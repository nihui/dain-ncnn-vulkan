# DAIN ncnn Vulkan

An ncnn implementation of DAIN, Depth-Aware Video Frame Interpolation.

dain-ncnn-vulkan uses the [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## Download

Download the Windows/Linux/MacOS executable for Intel/AMD/Nvidia GPUs:

**https://github.com/nihui/dain-ncnn-vulkan/actions**

Packages include all the binaries and models required. It is portable, so no CUDA or Caffe runtime environment is needed :)

## About DAIN

DAIN (Depth-Aware Video Frame Interpolation) (CVPR 2019)

https://github.com/baowenbo/DAIN

Wenbo Bao, Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang

This work is developed based on our TPAMI work MEMC-Net, where we propose the adaptive warping layer. Please also consider referring to it.

https://sites.google.com/view/wenbobao/dain

http://arxiv.org/abs/1904.00830

## Usage

Input two frame images, output one interpolated frame image.

### Example Command

```shell
./dain-ncnn-vulkan -0 0.jpg -1 1.jpg -o 01.jpg
./dain-ncnn-vulkan -i input_frames/ -o output_frames/
```

### Video Interpolation with FFmpeg

```shell
mkdir input_frames
mkdir output_frames

# find the source fps and format with ffprobe, for example 24fps, AAC
ffprobe input.mp4

# extract audio
ffmpeg -i input.mp4 -vn -acodec copy audio.m4a

# decode all frames
ffmpeg -i input.mp4 input_frames/frame_%06d.png

# interpolate 2x frame count
./dain-ncnn-vulkan -i input_frames -o output_frames

# encode interpolated frames in 48fps with audio
ffmpeg -framerate 48 -i output_frames/%06d.png -i audio.m4a -c:a copy -crf 20 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Full Usage

```console
Usage: dain-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...
       dain-ncnn-vulkan -i indir -o outdir [options]...

  -h                   show this help
  -v                   verbose output
  -0 input0-path       input image0 path (jpg/png/webp)
  -1 input1-path       input image1 path (jpg/png/webp)
  -i input-path        input image directory (jpg/png/webp)
  -o output-path       output image path (jpg/png/webp) or directory
  -n num-frame         target frame count (default=N*2)
  -s time-step         time step (0~1, default=0.5)
  -t tile-size         tile size (>=128, default=256) can be 256,256,128 for multi-gpu
  -m model-path        dain model path (default=best)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -f pattern-format    output image filename pattern format (%08d.jpg/png/webp, default=ext/%08d.png)
```

- `input0-path`, `input1-path` and `output-path` accept file path
- `input-path` and `output-path` accept file directory
- `num-frame` = target frame count
- `time-step` = interpolation time
- `tile-size` = tile size, use a smaller value to reduce GPU memory usage, it must be multiple of 32, default is 256
- `load:proc:save` = thread count for the three stages (image decoding + dain interpolation + image encoding), using larger values may increase GPU usage and consume more GPU memory. You can tune this configuration with "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.
- `pattern-format` = the filename pattern and format of the image to be output, png is better supported, however webp generally yields smaller file sizes, both are losslessly encoded

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Building from Source

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can also get the essential build requirements from your package manager
 
Fedora/CentOS/RHEL:
```shell
dnf install vulkan-headers vulkan-loader-devel
```
Debian/Ubuntu
```shell
apt-get install libvulkan-dev
```
Arch/Manjaro
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

* test-time sptial augmentation aka TTA-s
* test-time temporal augmentation aka TTA-t

## Original DAIN Project

- https://github.com/baowenbo/DAIN

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network interface on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding PNG/JPEG images on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directories on Windows
