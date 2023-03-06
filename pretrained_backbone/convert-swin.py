#!/usr/bin/env python
import sys
import torch

"""
Usage:
  # download SwinTransformer (tiny or small version) from:
  https://github.com/microsoft/Swin-Transformer
  
  # run the conversion, for example:
  ./convert_swin.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224_convert.pth

  # Then, use the weights with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224_convert.pth"
  
"""

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    source_weights = torch.load(input, map_location="cpu")['model']
    converted_weights = {}
    keys = list(source_weights.keys())

    for key in keys:
        new_key = 'detection_transformer.backbone.0.backbone.' + key
        converted_weights[new_key] = source_weights[key]

    torch.save(converted_weights, output)