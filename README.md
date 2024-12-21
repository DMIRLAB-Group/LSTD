# LSTD: Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting (AAAI-25)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

:triangular_flag_on_post:**News**(Dec 26, 2024): After the meeting, we will upload this paper to arXiv.

## Motivation
In the industry, since time series data often arrives sequentially and is accompanied by temporal distribution shifts. We observe that nonstationarity is brought by the unknown interventions on short-term states. Moreover, to address the online forecasting task, it is intuitive to find that we should disentangle the long/short-term states from the time series with unknown
interventions
<p align="center">
<img src=".\LSTD-main\Image\intervention.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> llustration of sequentially arriving exchange rate data, which is influenced by short-term customs duties and long-
term financial revenue. Moreover, the short-term customs duties are intervened by sudden customs tariff policies.
</p>

## Model
- To preserve the long-term dependencies in the long-term latent variables, we propose the smooth constraint. $A_{z_h^s}$ and $A_{z_e^s}$ denote the association matrices of the start half and the end half segments, hence we can restrict the long-term dependencies by restricting the similarity of these two matrices.
- We propose the interrupted dependency constraint for the short-term variables. Since the nonstationarity is assumed to be led by the interventions to the short-term latent variables, given $z_{1:H}^d$, if intervention occurs at $\tau$-th time step, and $2<\tau<H-1$, then $\frac{\partial \varepsilon_{H, i}^d}{\partial z_{\tau-1,j}^d}=0$, where $i,j \in { 1,\cdots,n_d}$, hence we can restrict the short-term dependencies by restricting the sparsity of the matrice.
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
python loop_LSTD.py -seed $seed -dataset $dataset -len $1/24/60
```
Multiple seeds and datasets can be run at one time.The important parameters are in file LSTD_config.py and you can go inside to change the parameters you want.

More parameter information please refer to `main.py`.


## <span id="resultslink">Results</span>

<p align="center">
<img src="./img/result_univariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 4.</b> Univariate forecasting results.
</p>

<p align="center">
<img src="./img/result_multivariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 5.</b> Multivariate forecasting results.
</p>


## FAQ
If you run into a problem like `RuntimeError: The size of tensor a (98) must match the size of tensor b (96) at non-singleton dimension 1`, you can check torch version or modify code about `Conv1d` of `TokenEmbedding` in `models/embed.py` as the way of circular padding mode in Conv1d changed in different torch versions.


## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following papers:

```
@article{haoyietal-informerEx-2023,
  author    = {Haoyi Zhou and
               Jianxin Li and
               Shanghang Zhang and
               Shuai Zhang and
               Mengyi Yan and
               Hui Xiong},
  title     = {Expanding the prediction capacity in long sequence time-series forecasting},
  journal   = {Artificial Intelligence},
  volume    = {318},
  pages     = {103886},
  issn      = {0004-3702},
  year      = {2023},
}
```
```
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

## Contact
If you have any questions, feel free to contact Haoyi Zhou through Email (zhouhaoyi1991@gmail.com) or Github issues. Pull requests are highly welcomed!

## Acknowledgments
Thanks for the computing infrastructure provided by Beijing Advanced Innovation Center for Big Data and Brain Computing ([BDBC](http://bdbc.buaa.edu.cn/)).
At the same time, thank you all for your attention to this work! [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhouhaoyi%2FInformer2020&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits+Count&edge_flat=false)](https://hits.seeyoufarm.com)
[![Stargazers repo roster for @zhouhaoyi/Informer2020](https://reporoster.com/stars/zhouhaoyi/Informer2020)](https://github.com/zhouhaoyi/Informer2020/stargazers)
