
# **End-to-end fusion of RGB and low-resolution NIR for enhanced crop-specific NIR generation**
> Md Yearat Hossain and Soumyabrata Dev

## Abstract

Near-infrared (NIR) imaging plays a crucial role in applications such as vegetation monitoring, remote sensing, and environmental analysis. However, NIR imaging systems are often expensive and limited in resolution compared to conventional RGB sensors. To address this limitation, we propose an end-to-end deep learning framework that generates high-resolution (HR) NIR images by fusing high-resolution RGB images with low-resolution (LR) NIR inputs. The framework integrates a lightweight image-to-image (I2I) backbone for extracting rich spatial features from the RGB domain, dual feature extractors operating in a multi-scale paradigm, and a learnable fusion module that adaptively combines spatial and spectral information. A decoder network then reconstructs the fused features into the final HR NIR output. Extensive experiments on paired aerial RGB--NIR crop datasets demonstrate that the proposed method significantly outperforms standalone RGB-to-NIR translation and conventional LR-to-HR NIR super-resolution approaches, improving performance from approximately 27.6dB to over 34.2dB Peak Signal-to-Noise Ratio (PSNR) and from 0.90 to 0.95 Structural Similarity Index (SSIM) when fusing 128x128 NIR inputs. Notably, even when the NIR input is 32 times lower in resolution (16x16), the framework consistently surpasses RGB-only baselines, achieving around 29.6dB PSNR and 0.91 SSIM. Beyond quantitative evaluation, additional analyses including SSIM error map visualization, cross-resolution and cross-dataset testing, and reconstruction variability assessment provide deeper insight into the robustness, limitations, and behavior of the framework under challenging conditions. The modular design of the architecture further enables flexible adaptation to a broad range of multi-modal and cross-spectral imaging tasks.

## Setup
The follwoing repository is based on the Pytorch implementation of LYT-Net which can be found here: [https://github.com/albrateanu/LYT-Net]

- Make Conda Environment

If first time using conda run:
```bash
conda init
```

then
```bash
conda create -n LYT_Torch python=3.9 -y
conda activate LYT_Torch
```

- Install Dependencies

```bash

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm torchmetrics

```

### Prepare Datasets


<summary>

<b>Datasets should be organized as follows:</b>

</summary>

  

```

|--dataset

| |--Canola

| | |--Train

| | | |--RGB

| | | | ...

| | | |--NIR

| | | | ...

| | |--Test

| | | |--RGB

| | | | ...

| | | |--NIR

| | | | ...

```

### Train

```bash

python train_fuse.py

```

### Test

You can test the model using the following commands. Pre-trained weights are available at ```trained_weights``` folder.




```bash

python test_fuse.py

```



**Note:** Please modify the dataset and trained weight paths in ```test_fuse.py``` as per your requirements.
