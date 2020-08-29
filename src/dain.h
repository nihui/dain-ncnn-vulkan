// dain implemented with ncnn library

#ifndef DAIN_H
#define DAIN_H

#include <string>

// ncnn
#include "net.h"

class DAIN
{
public:
    DAIN();
    ~DAIN();

    int load();

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, ncnn::Mat& outimage) const;

public:
    // dain parameters
    int prepadding;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net depthnet;
    ncnn::Net flownet;
    ncnn::Net ctxnet;
    ncnn::Net interpolation;
    ncnn::Pipeline* dain_preproc;
    ncnn::Pipeline* dain_postproc;
};

#endif // DAIN_H
