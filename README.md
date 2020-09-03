# DAIN ncnn Vulkan

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

ncnn implementation of DAIN, Depth-Aware Video Frame Interpolation.

dain-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## [Download]

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia GPU

**https://github.com/nihui/dain-ncnn-vulkan/actions**

This package includes all the binaries and models required. It is portable, so no CUDA or Caffe runtime environment is needed :)

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

### Full Usages

```console
Usage: dain-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...

  -h                   show this help
  -0 input0-path       input image0 path (jpg/png)
  -1 input1-path       input image1 path (jpg/png)
  -o output-path       output image path (jpg/png)
  -s time-step         time step (0~1, default=0.5)
  -t tile-size         tile size (>=128, default=256)
  -g gpu-id            gpu device to use (default=0)
```

- `input0-path`, `input1-path` and `output-path` accept file path
- `time-step` = interpolation time
- `tile-size` = tile size, use smaller value to reduce GPU memory usage, must be multiple of 32, default 256

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/nihui/dain-ncnn-vulkan.git
cd dain-ncnn-vulkan
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
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
