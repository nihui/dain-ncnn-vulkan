# dain-ncnn-vulkan

This software is in the early development stage, it may bite your cat  :exclamation:

Input two frame images, output one interpolated frame image.

```bash
./dain-ncnn-vulkan -i 0.jpg,1.jpg -o 01.jpg
```

### TODO
* interpolate rate control
* add more DAIN models
* port all custom layers to vulkan
* port pre-process and post-process to vulkan
* tiled process for reducing VRAM
* test-time sptial augmentation aka TTA-s
* test-time temporal augmentation aka TTA-t
* load images from directory
* read write webp
* good multi-gpu support
