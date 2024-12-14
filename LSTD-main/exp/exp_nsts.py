from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.nsts import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
from scipy.stats import norm
import numpy as np
from einops import rearrange
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import MultiheadAttention


from functorch import vmap, jacfwd, grad
import torch.distributions as D

import os
import time
from pathlib import Path

import warnings

warnings.filterwarnings('ignore')


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]


class Embedding_Net(nn.Module):

    def __init__(self, patch_size, input_len, out_len, emb_dim) -> None:
        super().__init__()
        self.patch_size = patch_size if patch_size <= input_len else input_len
        self.stride = self.patch_size // 2
        self.out_len = out_len

        self.num_patches = int((input_len - self.patch_size) / self.stride + 1)

        self.net1 = MLP(1, in_dim=self.patch_size, out_dim=emb_dim)
        self.net2 = MLP(1, emb_dim * self.num_patches, out_dim=self.out_len)

    def forward(self, x):
        B, L, M = x.shape
        if self.num_patches != 1:
            x = rearrange(x, 'b l m -> b m l')
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, 'b m n p -> (b m) n p')
        else:
            x = rearrange(x, 'b l m -> (b m) 1 l')
        x = self.net1(x)
        outputs = self.net2(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b  l m', b=B)
        return outputs


class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_norm - 2):
                net.append(nn.Linear(in_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Base_Net(nn.Module):
    def __init__(self, input_len, out_len, hidden_dim, input_dim, out_dim, is_mean_std=True, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len
        self.c_type = c_type
        self.out_dim = out_dim
        self.c_type = "type1" if out_dim != input_dim and c_type == "None" else c_type
        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=hidden_dim,
                           activation=activation,
                           layer_norm=layer_norm)
        elif self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=hidden_dim,
                           out_dim=self.out_dim * self.radio,
                           layer_norm=layer_norm, activation=activation)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=hidden_dim,
                           activation=activation,
                           out_dim=self.out_dim * input_len * self.radio, layer_norm=layer_norm)

        self.dropout_net = nn.Dropout(drop_out)

    def forward(self, x):
        if self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim)

        elif self.c_type == "None":
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout_net(x)
        if self.radio == 2:
            dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
            x = torch.chunk(x, dim=dim, chunks=2)
        return x


class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NPTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)

        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = args.hidden_layers
        self.dropout = args.dropout
        self.depth = args.depth
        self.activation = args.activation
        self.lags = args.lags
        self.stationary_transition_prior = NPTransitionPrior(lags=self.args.lags,
                                                             latent_size=self.args.enc_in,
                                                             num_layers=1,
                                                             hidden_dim=self.args.hidden_dim).to(self.device)
        if args.mode == 'time':
            encoder = TSEncoder(
                args,
                input_dims=args.seq_len,
                output_dims=320,  # standard ts2vec backbone value
                hidden_dims=64,  # standard ts2vec backbone value
                depth=args.depth)
            self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
            # self.zs_std = nn.Linear(args.seq_len, 320).to(self.device)

            self.regressor = Base_Net(2 * self.args.pred_len, self.args.pred_len, self.hidden_dim, args.enc_in, args.enc_in,
                                      is_mean_std=False, activation=args.activation,
                                      layer_norm=False, c_type="None", drop_out=self.dropout,
                                      layer_nums=self.hidden_layers).to(self.device)

            self.zs_rec = nn.Linear(320, 2 * args.seq_len).to(self.device)
            self.x_rec = nn.Linear(2 * args.seq_len, args.seq_len).to(self.device)
            self.zs_pred = nn.Linear(args.seq_len, 2 * args.pred_len).to(self.device)
            # self.zs_pred = Base_Net(args.seq_len, 2 * args.pred_len, self.hidden_dim, args.enc_in, args.enc_in,
            #                        is_mean_std=False, activation="gelu",
            #                          layer_norm=True, c_type="None", drop_out=0, layer_nums=2).to(self.device)

            self.x_pred = nn.Linear(args.pred_len, args.pred_len).to(self.device)

            # self.zd_rec = nn.Linear(self.args.seq_len, 2 * args.seq_len).to(self.device)
            self.zd_rec = Base_Net(self.args.seq_len, 2 * args.seq_len, self.hidden_dim, args.enc_in, args.enc_in,
                                   is_mean_std=False, activation=args.activation,
                                   layer_norm=True, c_type="None", drop_out=self.dropout, layer_nums=self.hidden_layers).to(
                self.device)

            self.zd_pred = nn.Linear(self.args.seq_len, 2 * args.pred_len).to(self.device)
            self.attention = MultiheadAttention(embed_dim=args.enc_in, num_heads=1).to(self.device)
        else:
            encoder = TSEncoder(
                args,
                input_dims=args.enc_in,
                output_dims=320,  # standard ts2vec backbone value
                hidden_dims=64,  # standard ts2vec backbone value
                depth=args.depth)
            self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
            # self.zs_std = nn.Linear(args.seq_len, 320).to(self.device)

            self.regressor = Base_Net(2 * self.args.enc_in, self.args.enc_in, self.hidden_dim, 2 * args.enc_in,
                                      args.enc_in,
                                      is_mean_std=False, activation=args.activation,
                                      layer_norm=False, c_type="type1", drop_out=self.dropout,
                                      layer_nums=self.hidden_layers).to(self.device)

            self.zs_rec = nn.Linear(320, 2 * args.enc_in).to(self.device)
            self.x_rec = nn.Linear(2 * args.enc_in, args.enc_in).to(self.device)
            self.zs_pred = nn.Linear(args.seq_len, 2 * args.pred_len).to(self.device)

            self.x_pred = nn.Linear(args.pred_len, args.pred_len).to(self.device)

            # self.zd_rec = nn.Linear(self.args.seq_len, 2 * args.seq_len).to(self.device)
            self.zd_rec = Base_Net(self.args.enc_in, 2 * args.enc_in, self.hidden_dim, self.args.enc_in,
                                   2 * args.enc_in,
                                   is_mean_std=False, activation=args.activation,
                                   layer_norm=True, c_type="type1", drop_out=self.dropout,
                                   layer_nums=self.hidden_layers).to(
                self.device)

            self.zd_pred = nn.Linear(self.args.seq_len, 2 * args.pred_len).to(self.device)

            self.attention = MultiheadAttention(embed_dim=args.enc_in, num_heads=1).to(self.device)

        self.register_buffer('stationary_dist_mean', torch.zeros(self.args.enc_in).to(self.device))
        self.register_buffer('stationary_dist_var', torch.eye(self.args.enc_in).to(self.device))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, is_training, return_feature=False):
        if self.args.mode == 'time':
            zs_ = self.encoder.encoder.forward_time(x)
            zs_ = self.zs_rec(zs_)
            zd_ = self.zd_rec(x.float()).permute(0, 2, 1)
            # print(zs_mean.shape)
            zs_rec_mean, zs_rec_std = zs_[:, :, :self.args.seq_len], self.sigmoid(zs_[:, :, -self.args.seq_len:])
            zd_rec_mean, zd_rec_std = zd_[:, :, :self.args.seq_len], self.sigmoid(zd_[:, :, -self.args.seq_len:])
            # print(zs_std.shape)
            if is_training:
                # zs_std =  self.zs_std(x.permute(0, 2, 1).float())
                # zs_std = self.sigmoid(zs_std)
                zs_rec = self.__reparametrize(zs_rec_mean, zs_rec_std)
                zd_rec = self.__reparametrize(zd_rec_mean, zd_rec_std)
            else:
                zs_rec = zs_rec_mean
                zd_rec = zd_rec_mean
            zs_pred = self.zs_pred(zs_rec)
            zd_pred = self.zd_pred(zd_rec)
            zs_pred_mean, zs_pred_std = zs_pred[:, :, :self.args.pred_len], self.sigmoid(zs_[:, :, -self.args.pred_len:])
            zd_pred_mean, zd_pred_std = zd_pred[:, :, :self.args.pred_len], self.sigmoid(zd_[:, :, -self.args.pred_len:])
            if is_training:
                # zs_pred_std =  self.zs_std(x.permute(0, 2, 1).float())
                # zs_std = self.sigmoid(zs_std)
                zs_pred_rec = self.__reparametrize(zs_pred_mean, zs_pred_std)
                zd_pred_rec = self.__reparametrize(zd_pred_mean, zd_pred_std)
            else:
                zs_pred_rec = zs_pred_mean
                zd_pred_rec = zd_pred_mean
            x_rec = self.x_rec(torch.cat([zs_rec, zd_rec], dim=-1)).transpose(1, 2)
            # y = self.regressor(zs_pred_rec).transpose(1, 2)

            y = self.regressor(torch.cat([zs_pred_rec, zd_pred_rec], dim=-1).transpose(1, 2))
            zs_q1 = zs_rec[:, :, :(self.args.seq_len // 2)].permute(0, 2, 1)
            zs_q2 = zs_rec[:, :, -(self.args.seq_len // 2):].permute(0, 2, 1)
        else:
            zs_ = self.encoder.encoder.forward_time(x.permute(0, 2, 1))  # B * T *  320
            zs_ = self.zs_rec(zs_)  # B * T *(2* dim)
            zd_ = self.zd_rec(x.float())  # B * T * (2*dim)
            # print(zs_mean.shape)
            zs_rec_mean, zs_rec_std = zs_[:, :, :self.args.enc_in], self.sigmoid(zs_[:, :, -self.args.enc_in:])
            zd_rec_mean, zd_rec_std = zd_[:, :, :self.args.enc_in], self.sigmoid(zd_[:, :, -self.args.enc_in:])
            # print(zs_std.shape)
            if is_training:
                # zs_std =  self.zs_std(x.permute(0, 2, 1).float())
                # zs_std = self.sigmoid(zs_std)
                zs_rec = self.__reparametrize(zs_rec_mean, zs_rec_std)  # B * T * dim
                zd_rec = self.__reparametrize(zd_rec_mean, zd_rec_std)  # B * T * dim
            else:
                zs_rec = zs_rec_mean
                zd_rec = zd_rec_mean
            zs_pred = self.zs_pred(zs_rec.permute(0, 2, 1))  # B * dim * 2T
            zd_pred = self.zd_pred(zd_rec.permute(0, 2, 1))  # B * dim * 2T
            zs_pred_mean, zs_pred_std = zs_pred[:, :, :self.args.pred_len].permute(0, 2, 1), self.sigmoid(
                zs_pred[:, :, -self.args.pred_len:].permute(0, 2, 1))
            zd_pred_mean, zd_pred_std = zd_pred[:, :, :self.args.pred_len].permute(0, 2, 1), self.sigmoid(
                zd_pred[:, :, -self.args.pred_len:].permute(0, 2, 1))  # B * T * dim
            if is_training:
                # zs_pred_std =  self.zs_std(x.permute(0, 2, 1).float())
                # zs_std = self.sigmoid(zs_std)
                zs_pred_rec = self.__reparametrize(zs_pred_mean, zs_pred_std)
                zd_pred_rec = self.__reparametrize(zd_pred_mean, zd_pred_std)
            else:
                zs_pred_rec = zs_pred_mean
                zd_pred_rec = zd_pred_mean
            x_rec = self.x_rec(torch.cat([zs_rec, zd_rec], dim=-1))
            # y = self.regressor(zs_pred_rec).transpose(1, 2)

            y = self.regressor(torch.cat([zs_pred_rec, zd_pred_rec], dim=-1))

            zs_q1 = zs_rec[:, :(self.args.seq_len // 2), :]
            zs_q2 = zs_rec[:, -(self.args.seq_len // 2):, :]


        output1, weights1 = self.attention(zs_q1, zs_q1, zs_q1)
        output2, weights2 = self.attention(zs_q2, zs_q2, zs_q2)
        L2_loss = torch.mean((weights1 - weights2) ** 2)
        for name, param in self.zd_pred.named_parameters():
            L1_loss = torch.abs(param).sum()
        if is_training:
            if self.args.mode == 'time':
                zs_kl_loss = self.kl_loss(torch.cat([zs_rec_mean, zs_pred_mean], dim=2).permute(0, 2, 1),
                                          torch.cat([zs_rec_std, zs_pred_std], dim=2).permute(0, 2, 1),
                                          torch.cat([zs_rec, zs_pred_rec], dim=2).permute(0, 2, 1))

                zd_kl_loss = self.kl_loss(torch.cat([zd_rec_mean, zd_pred_mean], dim=2).permute(0, 2, 1),
                                          torch.cat([zd_rec_std, zd_pred_std], dim=2).permute(0, 2, 1),
                                          torch.cat([zd_rec, zd_pred_rec], dim=2).permute(0, 2, 1))

                other_loss = zs_kl_loss * self.args.zc_kl_weight + zd_kl_loss * self.args.zd_kl_weight + self.args.L1_weight * L1_loss + self.args.L2_weight * L2_loss
            else:
                zs_kl_loss = self.kl_loss(torch.cat([zs_rec_mean, zs_pred_mean], dim=1),
                                          torch.cat([zs_rec_std, zs_pred_std], dim=1),
                                          torch.cat([zs_rec, zs_pred_rec], dim=1))

                zd_kl_loss = self.kl_loss(torch.cat([zd_rec_mean, zd_pred_mean], dim=1),
                                          torch.cat([zd_rec_std, zd_pred_std], dim=1),
                                          torch.cat([zd_rec, zd_pred_rec], dim=1))

                other_loss = zs_kl_loss * self.args.zc_kl_weight + zd_kl_loss * self.args.zd_kl_weight + self.args.L1_weight * L1_loss + self.args.L2_weight * L2_loss
        else:
            other_loss = self.args.L1_weight * L1_loss + self.args.L2_weight * L2_loss

        return x_rec, rearrange(y, 'b t d -> b (t d)'), other_loss

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.stationary_transition_prior(z_est)
        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()
        loss = (kld_normal + kld_laplace)
        return loss

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device=self.device)

        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                            .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            delay_fb=args.delay_fb,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                x_rec, pred, true, other_loss = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true) + criterion(x_rec.to(self.device).float(),
                                                         batch_x.to(self.device).float()) + other_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                self.model.store_grad()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            x_rec, pred, true, other_loss = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):

        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        preds = []
        trues = []
        x_list = []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            x_rec, pred, true, other_loss = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.reshape(self.args.pred_len, self.args.enc_in).detach().cpu())  # (1, pred_len * enc_ibn)
            trues.append(true.reshape(self.args.pred_len, self.args.enc_in).detach().cpu())
            x_list.append(torch.squeeze(batch_x).detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
        # os.makedirs(f"./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/", exist_ok=True)
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/pred.npy', np.array(preds))
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/y.npy', np.array(trues))
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/x.npy', np.array(x_list))
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)

        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode == 'test':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                x_rec, outputs, other_loss = self.model(x, True)
        else:
            x_rec, outputs, other_loss = self.model(x, True)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return x_rec.to(self.device).float(), outputs, rearrange(batch_y, 'b t d -> b (t d)'), other_loss



    def plt_mask(self, x, mask):
        import matplotlib.pyplot as plt
        x, mask = x[0].cpu(), mask[0].cpu()
        # 绘制折线图
        plt.plot(x, label='x')

        # 标注 mask=True 的位置
        mask_indices = torch.nonzero(mask).squeeze().numpy()
        plt.scatter(mask_indices, x[mask_indices], color='red', label='mask=True')

        # 添加标签和图例
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Line Plot with Mask')
        plt.legend()
        plt.savefig('mask.pdf')
        plt.close()

    def _ol_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()

        x = batch_x.to(self.device)
        batch_y = batch_y.float().to(self.device)
        for _ in range(self.n_inner):
            if self.online == 'none':
                with torch.no_grad():
                    x_rec, outputs, other_loss = self.model(x, False)
            else:
                x_rec, outputs, other_loss = self.model(x, False)
                loss = criterion(outputs, true) + criterion(x_rec.to(self.device).float(), batch_x.to(self.device).float()) + other_loss
                loss.backward()
                self.opt.step()
                self.model.store_grad()
                self.opt.zero_grad()

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        torch.cuda.empty_cache()
        return x_rec, outputs, rearrange(batch_y, 'b t d -> b (t d)'), other_loss
