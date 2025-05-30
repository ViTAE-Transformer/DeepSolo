<h1 align="center">DeepSolo & DeepSolo++</h1> 

<p align="center">
  <a href="#News">News</a> |
  <a href="#Usage">Usage</a> |
  <a href="#Citation">Citation</a> |
  <a href="#Acknowledgement">Acknowledgement</a>
</p >
This is the official repo for the papers:

> [**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**](https://arxiv.org/abs/2211.10772)
> 
> [**DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting**](https://arxiv.org/abs/2305.19957)

<img src="./figs/DeepSolo.jpg" alt="image" style="zoom:50%;" />

## News

- **[`2025/05/16`]**: :rocket::rocket: We release [LogicOCR](https://github.com/MiliLab/LogicOCR), a benchmark designed to evaluate the logical reasoning abilities of Large Multimodal Models (LMMs) on text-rich images, while minimizing reliance on domain-specific knowledge. We offer key insights for enhancing multimodal reasoning.

- **[`2024/04/25`]**: Update DeepSolo models finetuned on BOVText and DSText video datasets. 

- **[`2023/06/02`]**: Update the pre-trained and fine-tuned Chinese scene text spotting model (78.3% 1-NED on ICDAR 2019 ReCTS). 

- **[`2023/05/31`]**: The extension paper (DeepSolo++) is submitted to ArXiv. The code and models will be released soon.

- **[`2023/02/28`]**: DeepSolo is accepted by CVPR 2023. :tada::tada:

***

Relevant Project: 

> :sparkles: [**LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images?** ](https://arxiv.org/abs/2505.12307) | [Project Page](https://ymy-k.github.io/LogicOCR.github.io/)
> 
> :sparkles: [**Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation** ](https://arxiv.org/abs/2401.17904) | [Code](https://github.com/ymy-k/Hi-SAM)
> 
> :sparkles: [**GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching** ](https://arxiv.org/abs/2401.07080) | [Code](https://github.com/Hxyz-123/GoMatching)
> 
> [**DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer** ](https://arxiv.org/abs/2207.04491) | [Code](https://github.com/ymy-k/DPText-DETR)


Other applications of [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer) inlcude: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | [Remote Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) | [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) | [Video Object Segmentation](https://github.com/ViTAE-Transformer/VOS-LLB)


## Usage

See README for [DeepSolo](./DeepSolo/README.md) and [DeepSolo++](./DeepSolo++/README.md) 

## Citation

If you find DeepSolo helpful, please consider giving this repo a star :star: and citing:
```bibtex
@inproceedings{ye2023deepsolo,
  title={DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19348--19357},
  year={2023}
}

@article{ye2023deepsolo++,
  title={DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={arxiv preprint arXiv:2305.19957},
  year={2023}
}
```

## Acknowledgement

This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet). For academic use, this project is licensed under the 2-clause BSD License.