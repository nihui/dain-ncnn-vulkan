# DAIN ncnn Vulkan

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

ncnn implementation of DAIN, Depth-Aware Video Frame Interpolation.

dain-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## About DAIN

DAIN (Depth-Aware Video Frame Interpolation) (CVPR 2019)

https://github.com/baowenbo/DAIN

Wenbo Bao, Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang

This work is developed based on our TPAMI work MEMC-Net, where we propose the adaptive warping layer. Please also consider referring to it.

https://sites.google.com/view/wenbobao/dain

http://arxiv.org/abs/1904.00830

## Usages

Input two frame images, output one interpolated frame image.

### Example Command

```shell
./dain-ncnn-vulkan -0 0.jpg -1 1.jpg -o 01.jpg
```

### TODO

* ~~interpolate rate control~~
* ~~port all custom layers to vulkan~~
* ~~port pre-process and post-process to vulkan~~
* ~~tiled process for reducing VRAM~~
* ~~github action ci~~
* test-time sptial augmentation aka TTA-s
* test-time temporal augmentation aka TTA-t
* load images from directory
* read write webp
* good multi-gpu support


## Original DAIN Project

- https://github.com/baowenbo/DAIN

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
