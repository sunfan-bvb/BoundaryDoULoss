# Boundary DoU Loss
This repo holds code for Boundary Difference Over Union Loss For Medical Image Segmentation(MICCAI 2023).

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Dataset
You can follow [TransUnet](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) to get and prepare the datasets.

2. The directory structure of the whole project is as follows:

```bash
.
├── TransUNet
│   └── 
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
├── Synapse
│   ├── test
│   │   ├── case0001.npy.h5
│   │   └── *.npy.h5
│   ├── train
│   │   ├── case0005_slice000.npz
│   │   └── *.npz
│   └── lists_Synapse
│       ├── all.lst
│       ├── test.txt
│       └── train.txt
└── ACDC
    └── ...(same as Synapse)
```

### 2. Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test
1. For Synapse dataset
* train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

* test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --vit_name R50-ViT-B_16 --is_savenii
```

2. For ACDC dataset
* train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset ACDC --vit_name R50-ViT-B_16
```

* test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset ACDC --vit_name R50-ViT-B_16 --is_savenii
```

## Results
Our results were trained and tested using five different seeds, with the final results being the average of the five runs. The seed settings and results for each run are shown in the table below. For example, for the ACDC dataset, we have

| Seed | Loss | mean dice | mean hd95 | boundary IoU| 
| - | :-: | -: | :-: | :-: |
| 1234 | Boundary DoU| 91.40 | 2.20 | 78.71 |
| 1111 | Boundary DoU | 91.22 | 2.41 | 78.04 |
| 2222 | Boundary DoU | 91.16 | 2.08 | 78.75 |
| 3333 | Boundary DoU | 91.41 | 2.00 | 78.33 |
| 4444 | Boundary DoU | 91.30 | 2.16 | 78.47 |
| mean | Boundary DoU | 91.30 | 2.17 | 78.46 |

In the TransUNet model, the impact of seed selection on the results varies for different datasets, and different seeds can be tried for better results.

## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)


