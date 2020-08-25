// dain implemented with ncnn library

#ifndef DAIN_OPS_H
#define DAIN_OPS_H

#include <vector>

// ncnn
#include "layer.h"

class Correlation : public ncnn::Layer
{
public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

class OpticalFlowWarp : public ncnn::Layer
{
public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

class DepthFlowProjection : public ncnn::Layer
{
public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

class FilterInterpolation : public ncnn::Layer
{
public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

#endif // DAIN_OPS_H
