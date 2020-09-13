// dain implemented with ncnn library

#ifndef DAIN_OPS_H
#define DAIN_OPS_H

#include <vector>

// ncnn
#include "layer.h"
#include "pipeline.h"

class Correlation : public ncnn::Layer
{
public:
    Correlation();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Layer* padding;
    ncnn::Pipeline* pipeline_correlation;
    ncnn::Pipeline* pipeline_correlation_pack4to1;
};

class OpticalFlowWarp : public ncnn::Layer
{
public:
    OpticalFlowWarp();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_opticalflowwarp;
    ncnn::Pipeline* pipeline_opticalflowwarp_pack4;
};

class DepthFlowProjection : public ncnn::Layer
{
public:
    DepthFlowProjection();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_depthflowprojection_zero;
    ncnn::Pipeline* pipeline_depthflowprojection_project;
    ncnn::Pipeline* pipeline_depthflowprojection_average;
    ncnn::Pipeline* pipeline_depthflowprojection_fillhole;
};

class FilterInterpolation : public ncnn::Layer
{
public:
    FilterInterpolation();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_filterinterpolation;
    ncnn::Pipeline* pipeline_filterinterpolation_pack4;
};

#endif // DAIN_OPS_H
