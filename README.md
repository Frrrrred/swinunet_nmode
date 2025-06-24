# MeSwin-UNet: Lightweight Medical Image Segmentation

**MeSwin-UNet** is an advanced framework for medical image segmentation that synergistically integrates hierarchical Swin Transformer encoders with neural memory ODE (nmODE) decoders. This repository provides the official implementation of the approach described in the paper: **MeSwin-UNet: Lightweight Medical Image Segmentation via Hierarchical Dual-Masking and Robust Neural Memory ODEs**.

## Key Innovations
1. **Robust nmODE Decoders**  
   Novel discretization schemes (FED/EAD/RKD) for boundary preservation and noise robustness
2. **Lightweight Hybrid Design**  
   48% parameter reduction via depthwise separable convolution and low-rank compression
3. **Hierarchical Dual-Masking SSL**  
   Self-supervised pretraining with multi-scale masked reconstruction for annotation efficiency

## Performance Highlights

​	For MeSwin-UNet (RKD), the performance is as follows.

| Dataset  | DSC (%) | mIoU (%) | Params (M) | FLOPs (G) |
| -------- | ------- | -------- | ---------- | --------- |
| ISIC2017 | 88.92   | 79.87    | 13.45      | 3.34      |
| ISIC2018 | 89.78   | 81.37    | 13.45      | 3.34      |
| BUSI     | 84.75   | 76.40    | 13.45      | 3.34      |

**Annotation Efficiency**: Achieves 87.32% Dice with only 60% annotations

## Installation

​	pytorch=2.4.1, torchvision=0.19.1, torchaudio=2.4.1

```bash
conda create -n meswin python=3.9
conda activate meswin
conda install torch torchvision torchaudio
pip install -r requirements.txt
```

## Dataset Preparation
### Supported Datasets
​	The [ISIC17](https://challenge.isic-archive.com/data/#2017), [ISIC18](https://challenge.isic-archive.com/data/#2018) and [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) datasets, divided into a 7:3 ratio. After downloading the datasets, you are supposed to resize their resolution to 512x512 for ISIC17 and ISIC18, 256x256 for BUSI. Then you are supposed to put them into './data/isic17/', './data/isic18/' and './data/busi/' , and the file format reference is as follows. (take the ISIC17 dataset as an example.)

```
dataset/
├── isic17/
│   ├── train/
│   │   ├── *.png
│   └── val/
│   │   ├── *.png
│   ├── mask/
│   │   ├── train/
│   │   │   ├── *.png
│   │   └── val/
└── └── └── └── *.png
```

## Training Pipeline
### 1. Self-Supervised Pretraining
```bash
python main_swinUnet_nmODE.py
```

### 2. Supervised Fine-tuning
```bash
python main_finetune.py
```

### 3.Obtain the outputs

​	After trianing, you could obtain the outputs in './output/'.

## Code Structure

```
MeSwin-UNet/
├── configs/                  # Model configurations about SSL
├── data/                     # Dataset loading utilities
├── dataset/                  # Model configurations
├── models/
│   ├── nmODE_block/          # All available nmODE blocks including FED/EAD/RKD implementations (the full version can be found in './New_nmODE_Decoders')
│   ├── build.py              # Schedule model
│   ├── nmode_decoder.py      # Decoder composed of nmODE blocks
│   ├── swin_transformer.py   
│   └── swin_unet_nmode.py    # Full architecture of pre-training model
├── config.py                 # Other model configurations
├── logger.py
├── lr_scheduler.py
├── main_swinUnet_nmODE.py    # SSL training script
├── main_finetune.py          # Supervised fine-tuning
├── optimizer.py
└── utils
```

In './New_nmODE_Decoders', put the model file in the models folder, to change the decoder to nmODE please refer to the provided UNet example.

## Acknowledgements

This implementation builds upon:
- [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)
- [nmODE](https://github.com/ODE-Transformer/nmODE)

---
**Note**: For commercial use, please consult the Apache 2.0 license terms in [LICENSE](LICENSE).