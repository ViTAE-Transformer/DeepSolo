<h1 align="center">DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting</h1> 

<p align="center">
<a href="https://arxiv.org/pdf/2211.10772.pdf"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Main Results">Main Results</a> |
</p >
*This is the official repo of the paper "DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting".*

## Introduction

**Abstract.** End-to-end text spotting aims to integrate scene text detection and recognition into a unified framework. Dealing with the relationship between the two sub-tasks plays a pivotal role in designing effective spotters. Although transformer-based methods eliminate the heuristic post-processing, they still suffer from the synergy issue between the sub-tasks and low training efficiency. In this paper, we present DeepSolo, a simple detection transformer baseline that lets a single Decoder with Explicit Points Solo for text detection and recognition simultaneously. Technically, for each text instance, we represent the character sequence as ordered points and model them with learnable explicit point queries. After passing a single decoder, the point queries have encoded requisite text semantics and locations and thus can be further decoded to the center line, boundary, script, and confidence of text via very simple prediction heads in parallel, solving the sub-tasks in text spotting in a unified framework. Besides, we also introduce a text-matching criterion to deliver more accurate supervisory signals, thus enabling more efficient training. Quantitative experiments on public benchmarks demonstrate that DeepSolo outperforms previous state-of-the-art methods and achieves better training efficiency. In addition, DeepSolo is also compatible with line annotations, which require much less annotation cost than polygons.

<img src="./figs/DeepSolo.jpg" alt="image" style="zoom:50%;" />

## Main Results

**Total-Text**
|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-None|E2E-Full|#Params|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K|93.86|82.11|87.59|78.83|86.15|42.5M|-|
|Res-50|Synth150K + MLT17 + IC13 + IC15|93.09|82.11|87.26|79.65|87.00|42.5M|-|
|Res-50|Synth150K + MLT17 + IC13 + IC15 + TextOCR|93.19|84.64|88.72|**82.54**|**88.72**|42.5M|-|
|Res-101|Synth150K + MLT17 + IC13 + IC15|93.20|83.51|88.09|80.12|87.14|61.5M|-|
|Swin-T|Synth150K + MLT17 + IC13 + IC15|92.77|83.51|87.90|79.66|87.05|43.1M|-|
|Swin-S|Synth150K + MLT17 + IC13 + IC15|93.72|84.24|88.73|81.27|87.75|64.4M|-|
|ViTAEv2-S|Synth150K + MLT17 + IC13 + IC15|92.57|85.50|**88.89**|81.79|88.40|33.7M|-|

**ICDAR 2015 (IC15)**

|Backbone|External Data|Det-P|Det-R|Det-F1|E2E-S|E2E-W|E2E-G|#Params|Weights|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|Res-50|Synth150K + Total-Text + MLT17 + IC13|92.79|87.39|**90.01**|86.84|81.90|76.93|42.5M|-|
|Res-50|Synth150K + Total-Text + MLT17 + IC13 + TextOCR|92.54|87.19|89.79|**87.95**|**83.46**|**79.08**|42.5M|-|

***

***Pre-trained Models***

|Backbone|Training Data|Weights|
|:------:|:------:|:------:|
|Res-50|Synth150K + Total-Text|-|
|Res-50|Synth150K + Total-Text + MLT17 + IC13 + IC15|-|
|Res-50|Synth150K + Total-Text + MLT17 + IC13 + IC15 + TextOCR|-|
|Res-101|Synth150K + Total-Text + MLT17 + IC13 + IC15|-|
|Swin-T|Synth150K + Total-Text + MLT17 + IC13 + IC15|-|
|Swin-S|Synth150K + Total-Text + MLT17 + IC13 + IC15|-|
|ViTAEv2-S|Synth150K + Total-Text + MLT17 + IC13 + IC15|-|