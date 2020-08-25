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

private:
    ncnn::Net depthnet;
    ncnn::Net flownet;
    ncnn::Net ctxnet;
    ncnn::Net interpolation;
};

#endif // DAIN_H
