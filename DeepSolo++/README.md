<h1 align="center">DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting</h1> 

<p align="center">
  <a href="#Checkpoints">Main Results</a> |
  <a href="#Usage">Usage</a>
</p >


## Checkpoints

|Version|Pretrained weights|Finetuned Weights|
|:------:|:------:|:------:|
|Res-50, routing, #1| [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcwYxWkWMGz6y4XFYQ?e=5HdB0S) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcwZNWXa6BwI-R6SAQ?e=kEOoBQ) (MLT19 Task4 H-mean: 50.3) |
|Res-50, routing, #3| [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcwbfVjpFnjmdIVtmQ?e=3nozWk) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcwaS_q-E7wP6tJIoQ?e=eY291i) (MLT19 Task4 H-mean: 51.2) |

## Usage

- ### Installation

Python 3.8 + PyTorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6)
```
# git clone https://github.com/ViTAE-Transformer/DeepSolo.git
# cd DeepSolo/DeepSolo++
conda create -n deepsolo++ python=3.8 -y
conda activate deepsolo++
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install -e detectron2
python setup.py build develop
```

- ### Data Preparation

Synthetic multilingual training images from [ICDAR 2019](https://rrc.cvc.uab.es/?ch=15&com=downloads): [Arabic](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9BcmFiaWMuemlw) | [Bangla](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9CYW5nbGEuemlw) | [Chinese](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9DaGluZXNlLnppcA==) | [Japanese](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9KYXBhbmVzZS56aXA=) | [Korean](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9Lb3JlYW4uemlw) | [Latin](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9MYXRpbi56aXA=) | [Hindi](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=15&f=aHR0cDovL3B0YWsuZmVsay5jdnV0LmN6L3B1YmxpY19kYXRhc2V0cy9TeW50VGV4dC9IaW5kaS56aXA=) 

Training images of ArT, LSVT, MLT19: [Link](https://github.com/aim-uofa/AdelaiDet/tree/master/datasets)

Training images of RCTW: [Link](https://rctw.vlrlab.net/dataset)

Testing images: [MLT19](https://rrc.cvc.uab.es/?ch=15&com=downloads) | [MLT17](https://rrc.cvc.uab.es/?ch=8&com=downloads)

All json files: [Link](https://1drv.ms/u/s!AimBgYV7JjTlgcwd0WYCX-AEg7N8Sw?e=gJcHxe)

*Some image files need to be renamed.* Organize them as follows (lexicon files are not listed here):

```
|- ./datasets
   |- Arabic
   |  |- train_images
   |  └  train.json
   |- Bangla
   |  |- train_images
   |  └  train.json
   |- Chinese
   |  |- train_images
   |  └  train.json
   |- Hindi
   |  |- train_images
   |  └  train.json
   |- Japanese
   |  |- train_images
   |  └  train.json
   |- Korean
   |  |- train_images
   |  └  train.json
   |- Latin
   |  |- train_images
   |  └  train.json
   |- RCTW
   |  |- train_images
   |  └  train.json
   |- ArT
   |  |- rename_artimg_train
   |  └  train.json
   |- LSVT
   |  |- rename_lsvtimg_train
   |  └  train.json
   |- mlt19
   |  |- train_images
   |  |- test_images
   |  |- mlt19_train.json
   |  └  mlt19_test.json
   |- mlt17
   |  |- test_images
   |  └  mlt17_test.json

```


- ### Training

Before training, download [DeepSolo](https://1drv.ms/u/s!AimBgYV7JjTlgcdu018Hx6GHAo-ZCQ?e=NkEQt6) and put it in `./pretrained_model` for initialization.

**1. Pre-train**

```
python tools/train_net.py --config-file configs/R_50/mlt19_multihead/pretrain.yaml --num-gpus 8
```

**2. Fine-tune**

```
python tools/train_net.py --config-file configs/R_50/mlt19_multihead/finetune.yaml --num-gpus 8
```


- ### Evaluation
```
python tools/train_net.py --config-file configs/R_50/mlt19_multihead/finetune.yaml --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```
**Note:** To conduct evaluation on ICDAR MLT 2019, you can directly submit the saved file under `output/R50/bs8_600k_synth-textocr-init/finetune` to the [official website](https://rrc.cvc.uab.es/?ch=15&com=mymethods&task=1) for evaluation.


- ### Visualization Demo
```
python demo/demo.py --config-file configs/R_50/mlt19_multihead/finetune.yaml --input ${IMAGES_FOLDER_OR_ONE_IMAGE_PATH} --output ${OUTPUT_PATH} --opts MODEL.WEIGHTS <MODEL_PATH>
```


## Citation

```bibtex
@article{ye2023deepsolo++,
  title={DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={arxiv preprint arXiv:2305.19957},
  year={2023}
}
```
