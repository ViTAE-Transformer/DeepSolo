<h1 align="center">DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting</h1> 

<p align="center">
<a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<p align="center">
  <a href="#News">News</a> |
  <a href="#Main Results">Main Results</a> |
  <a href="#Usage">Usage</a> |
  <a href="#Citation">Citation</a> |
  <a href="#Acknowledgment">Acknowledgement</a>
</p >

This is the official repo for the paper "DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting".

<img src="./figs/DeepSolo.jpg" alt="image" style="zoom:50%;" />

## News

`2023.02.28` DeepSolo is accepted by CVPR 2023 :tada::tada:

## Main Results

**Total-Text**
|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-None|E2E-Full|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K|93.9|82.1|87.6|78.8|86.2|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd3oqq103k359L2PQ?e=tkxgol)|
|Res-50|Synth150K+MLT17+IC13+IC15|93.1|82.1|87.3|79.7|87.0|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd2FhvW7pjuKs4iLQ?e=TqYdjG)|
|Res-50|Synth150K+MLT17+IC13+IC15+TextOCR|93.2|84.6|88.7|$\underline{\text{82.5}}$|$\underline{\text{88.7}}$|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd138p8HaXViFk-tw?e=r15pMR)|
|Res-101|Synth150K+MLT17+IC13+IC15|93.2|83.5|88.1|80.1|87.1|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd0wgXgTNJg3lD4qQ?e=wuOPfN)|
|Swin-T|Synth150K+MLT17+IC13+IC15|92.8|83.5|87.9|79.7|87.1|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd5mc12FlChwGCUig?e=Xjdtis)|
|Swin-S|Synth150K+MLT17+IC13 +C15|93.7|84.2|88.7|81.3|87.8|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd4Rn_bg8cOn-LwEg?e=dVqz7z)|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15|92.6|85.5|$\underline{\text{88.9}}$|81.8|88.4|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd8dztVae7RRLn6Ow?e=2GLRAs)|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15+TextOCR|92.9|87.4|**90.0**|**83.6**|**89.6**|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd6XGlbZ-I7WvGslQ?e=rrkXLx)|

**ICDAR 2015 (IC15)**

|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-S|E2E-W|E2E-G|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17+IC13|92.8|87.4|90.0|86.8|81.9|76.9|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdp6_LjerVYzoYORw?e=0ZuXgR)|
|Res-50|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.5|87.2|89.8|$\underline{\text{88.0}}$|$\underline{\text{83.5}}$|$\underline{\text{79.1}}$|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdonZXu6_JtW2QMuA?e=8BTzmi)|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13|93.7|87.3|**90.4**|87.5|82.8|77.7|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdrUOUheq2dw6FP-A?e=PYXbiY)|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.4|87.9|$\underline{\text{90.1}}$|**88.1**|**83.9**|**79.5**|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdqw1UUnbSAG4qoWA?e=Co1prY)|

**CTW1500**

|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-None|E2E-Full|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15|93.2|85.0|88.9|64.2|81.4|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdsiFgSz-FHgKepqQ?e=56gdHj)|

***

***Pre-trained Models for Total-Text & ICDAR 2015***

|Backbone|Training Data|Weights|
|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdxUY6EC18kIvb2HA?e=GSC8Cx)|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdwrhMu_5lyV3j3gg?e=90flAQ)|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdu018Hx6GHAo-ZCQ?e=NkEQt6)|
|Res-101|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdvwoL7Y1PSlNFMgw?e=APocIV)|
|Swin-T|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdzxsGdxKgUOoiIVA?e=7BxJhq)|
|Swin-S|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdyjP6PtQSliVdJLA?e=hHkIs4)|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd7KPBhro8LU9fLjA?e=gcpVZ2)|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcd9wi432uitMgTM-w?e=fjuJbm)|

***Pre-trained Models for CTW1500***

|Backbone|Training Data|Weights|
|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdtYzwEBGvOH6CiBw?e=trgKFE)|

## Usage

to be updated on 2023.03.07

## Citation

If you find DeepSolo helpful, please consider giving this repo a star:star: and citing:
```bibtex
@inproceedings{ye2022deepsolo,
  title={DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={CVPR},
  year={2023}
}
```

## Forthcoming

- [ ] DeepSolo v2

## Acknowledgment

This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet).