<h1 align="center">DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting</h1> 

<p align="center">
<a href="https://arxiv.org/pdf/2211.10772.pdf"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

This is the official repo for the paper "DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting".

<img src="./figs/DeepSolo.jpg" alt="image" style="zoom:50%;" />

## Main Results

**Total-Text**
|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-None|E2E-Full|#Params|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K|93.86|82.11|87.59|78.83|86.15|42.5M|-|
|Res-50|Synth150K+MLT17+IC13+IC15|93.09|82.11|87.26|79.65|87.00|42.5M|-|
|Res-50|Synth150K+MLT17+IC13+IC15+TextOCR|93.19|84.64|88.72|$\underline{\text{82.54}}$|$\underline{\text{88.72}}$|42.5M|-|
|Res-101|Synth150K+MLT17+IC13 +IC15|93.20|83.51|88.09|80.12|87.14|61.5M|-|
|Swin-T|Synth150K+MLT17+IC13+IC15|92.77|83.51|87.90|79.66|87.05|43.1M|-|
|Swin-S|Synth150K+MLT17+IC13 +C15|93.72|84.24|88.73|81.27|87.75|64.4M|-|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15|92.57|85.50|$\underline{\text{88.89}}$|81.79|88.40|33.7M|-|
|ViTAEv2-S|Synth150K+MLT17+IC13+IC15+TextOCR|92.89|87.35|**90.04**|**83.59**|**89.62**|33.7M|-|

**ICDAR 2015 (IC15)**

|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-S|E2E-W|E2E-G|#Params|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K+Total-Text+MLT17 +IC13|92.79|87.39|90.01|86.84|81.90|76.93|42.5M|-|
|Res-50|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.54|87.19|89.79|$\underline{\text{87.95}}$|$\underline{\text{83.46}}$|$\underline{\text{79.08}}$|42.5M|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13|93.65|87.29|**90.36**|87.52|82.81|77.68|33.7M|-|
|ViTAEv2-S|Synth150K+Total-Text+MLT17+IC13+TextOCR|92.36|87.92|$\underline{\text{90.08}}$|**88.14**|**83.91**|**79.48**|33.7M|-|

***

***Pre-trained Models***

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