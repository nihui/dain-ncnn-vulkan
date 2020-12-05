// dain implemented with ncnn library

#ifndef DAIN_H
#define DAIN_H

#include <string>

// ncnn
#include "net.h"

class DAIN
{
public:
    DAIN(int gpuid);
    ~DAIN();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

public:
    // dain parameters
    int tilesize;
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
