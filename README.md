# LSTD: Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting (AAAI-25)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

:triangular_flag_on_post:**News**(Dec 26, 2024): After the meeting, we will upload this paper to arXiv.

## Motivation
In the industry, since time series data often arrives sequentially and is accompanied by temporal distribution shifts. We observe that nonstationarity is brought by the unknown interventions on short-term states. Moreover, to address the online forecasting task, it is intuitive to find that we should disentangle the long/short-term states from the time series with unknown interventions as shown in Figure 1.
<p align="center">
<img src=".\LSTD-main\Image\intervention.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> llustration of sequentially arriving exchange rate data, which is influenced by short-term customs duties and long-
term financial revenue. The short-term customs duties are intervened by sudden customs tariff policies.
</p>

## Model
- To preserve the long-term dependencies in the long-term latent variables, we propose the smooth constraint. $A_{z_h^s}$ and $A_{z_e^s}$ denote the association matrices of the start half and the end half segments, hence we can restrict the long-term dependencies by restricting the similarity of these two matrices.
- We propose the interrupted dependency constraint for the short-term variables. Since the nonstationarity is assumed to be led by the interventions to the short-term latent variables, given $z_{1:H}^d$, if intervention occurs at $\tau$-th time step, and $2<\tau<H-1$, then $\frac{\partial \varepsilon_{H, i}^d}{\partial z_{\tau-1,j}^d}=0$, where $i,j \in { 1,\cdots,n_d}$, hence we can restrict the short-term dependencies by restricting the sparsity of the matrice.
- Our model overview is as shown in Figure 2.
<p align="center">
<img src=".\LSTD-main\Image\model.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> The framework of the proposed LSTD model. The long/short-term latent variables $z_{1:L}^d$ and $z_{1:L}^s$ are extracted from the encoder. And the latent transition module is used to estimated the $z_{L+1:H}^d$ and the $z_{L+1:H}^s$ from $z_{1:L}^d$ and $z_{1:L}^s$, respectively. The long-term and short-term prior networks are used to estimate the prior distributions.

## Requirements

- Python 3.8
- torch == 2.3.1
- numpy == 1.23.5
- pandas == 1.5.3
- einops == 0.4.0
- tqdm == 4.64.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

We have already put the datasets in the .\LSTD-main\data\ file.You just should unzip datasets1 and unzip datasets2 and it can be used directly. 
## Reproducibility

To easily reproduce the results you can run the following commands:
```
python run_LSTD.py -seed $seed -dataset $dataset -len $1/24/60
```
Multiple seeds and datasets can be run at one time.The important parameters are in file LSTD_config.py and you can go inside to change the parameters you want.

And we provide explanations for the important parameters:
| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| root_path      | The root path of the data file (defaults to `./data/ETT/`)    |
| data_path      | The data file name (defaults to `ETTh2.csv`)                  |
| features       | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate) |
| seq_len | Input sequence length of Informer encoder (defaults to 60) |
| label_len | Start token length of Informer decoder (defaults to 0) |
| pred_len | Prediction sequence length (defaults to 1) |
| des | exp description |
| itr | experiments times |
| test_bsz | Batch size in test |
| train_epochs | Epochs in train |
| online_learning | It is online learning or not |
| L1_weight | The weights of L1 regularization. |
| L2_weight | The weights of L2 regularization. |
| dropout | The magnitude of dropout. |
| zd_kl_weight | The weight of the prior for the short-term effect. |
| zs_kl_weight | The weight of the prior for the long-term effect. |

More parameter information please refer to `main.py`.


## <span id="resultslink">Results</span>

The main results are shown in table 1.

<p align="center">
<img src=".\LSTD-main\Image\results.png" height = "500" alt="" align=center />
<br><br>
</p>




## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following papers:

```
To be continued...
```
