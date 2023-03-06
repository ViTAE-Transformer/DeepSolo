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
|Res-50|Synth150K|93.9|82.1|87.6|78.8|86.2|-|
|Res-50|Synth150K+MLT17+IC13+IC15|93.1|82.1|87.3|79.7|87.0|-|
|Res-50|Synth150K+MLT17+IC13+IC15+TextOCR|93.2|84.6|88.7|$\underline{\text{82.5}}$|$\underline{\text{88.7}}$|-|
|Res-101|Synth150K+MLT17+IC13 +IC15|93.2|83.5|88.1|80.1|87.1|-|
|Swin-T|Synth150K+MLT17+IC13+IC15|92.8|83.5|87.9|79.7|87.1|-|
|Swin-S|Synth150K+MLT17+IC13 +C15|93.7|84.2|88.7|81.3|87.8|-|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15|92.6|85.5|$\underline{\text{88.9}}$|81.8|88.4|-|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15+TextOCR|92.9|87.4|**90.0**|**83.6**|**89.6**|-|

**ICDAR 2015 (IC15)**

|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-S|E2E-W|E2E-G|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17 +IC13|92.8|87.4|90.0|86.8|81.9|76.9|-|
|Res-50|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.5|87.2|89.8|$\underline{\text{88.0}}$|$\underline{\text{83.5}}$|$\underline{\text{79.1}}$|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13|93.7|87.3|**90.4**|87.5|82.8|77.7|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.4|87.9|$\underline{\text{90.1}}$|**88.1**|**83.9**|**79.5**|-|

**CTW1500**
|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-None|E2E-Full|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15||||64.2|81.4|-|

***

***Pre-trained Models for Total-Text & ICDAR 2015***

|Backbone|Training Data|Weights|
|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text|-|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15|-|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR|-|
|Res-101|Synth150K+Total-Text+MLT17+IC13+IC15|-|
|Swin-T|Synth150K+Total-Text+MLT17+IC13+IC15|-|
|Swin-S|Synth150K+Total-Text+MLT17+IC13+IC15|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+IC15|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR|-|

***Pre-trained Models for CTW1500***

|Backbone|Training Data|Weights|
|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17+IC13+IC15|-|

## Usage

## Citation

## Acknowledgment