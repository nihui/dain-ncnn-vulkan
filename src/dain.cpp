// dain implemented with ncnn library

#include "dain.h"

#include <algorithm>
#include <vector>

#include "dain_ops.h"

DEFINE_LAYER_CREATOR(Correlation)
DEFINE_LAYER_CREATOR(OpticalFlowWarp)
DEFINE_LAYER_CREATOR(DepthFlowProjection)
DEFINE_LAYER_CREATOR(FilterInterpolation)

DAIN::DAIN()
{
    vkdev = ncnn::get_gpu_device(0);
}

DAIN::~DAIN()
{
}

int DAIN::load()
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;

    depthnet.opt = opt;
    flownet.opt = opt;
    ctxnet.opt = opt;
    interpolation.opt = opt;

    depthnet.set_vulkan_device(vkdev);
    flownet.set_vulkan_device(vkdev);
    ctxnet.set_vulkan_device(vkdev);
    interpolation.set_vulkan_device(vkdev);

    flownet.register_custom_layer("dain.Correlation", Correlation_layer_creator);
    flownet.register_custom_layer("dain.OpticalFlowWarp", OpticalFlowWarp_layer_creator);

    interpolation.register_custom_layer("dain.DepthFlowProjection", DepthFlowProjection_layer_creator);
    interpolation.register_custom_layer("dain.FilterInterpolation", FilterInterpolation_layer_creator);

    depthnet.load_param("depthnet.param");
    depthnet.load_model("depthnet.bin");

    flownet.load_param("flownet.param");
    flownet.load_model("flownet.bin");

    ctxnet.load_param("ctxnet.param");
    ctxnet.load_model("ctxnet.bin");

    interpolation.load_param("interpolation.param");
    interpolation.load_model("interpolation.bin");

    return 0;
}

int DAIN::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, ncnn::Mat& outimage) const
{
    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = in0image.elempack;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = depthnet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    {
        // pad to 32n+32
        int w_padded = (w + 31) / 32 * 32 + 32 - w;
        int h_padded = (h + 31) / 32 * 32 + 32 - h;

        ncnn::Mat in0_origin = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB2BGR, w, h);
        ncnn::Mat in1_origin = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB2BGR, w, h);

        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        in0_origin.substract_mean_normalize(0, norm_vals);
        in1_origin.substract_mean_normalize(0, norm_vals);

        ncnn::Mat in0;
        ncnn::Mat in1;
        ncnn::copy_make_border(in0_origin, in0, h_padded/2, h_padded - h_padded/2, w_padded/2, w_padded - w_padded/2, ncnn::BORDER_REPLICATE, 0.f);
        ncnn::copy_make_border(in1_origin, in1, h_padded/2, h_padded - h_padded/2, w_padded/2, w_padded - w_padded/2, ncnn::BORDER_REPLICATE, 0.f);

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in0_gpu;
        ncnn::VkMat in1_gpu;
        {
            cmd.record_upload(in0, in0_gpu, opt);
            cmd.record_upload(in1, in1_gpu, opt);
        }

        // depthnet
        ncnn::VkMat depth0;
        ncnn::VkMat depth1;
        {
            ncnn::Extractor ex = depthnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input", in0_gpu);
            ex.extract("depth", depth0, cmd);
        }
        {
            ncnn::Extractor ex = depthnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input", in1_gpu);
            ex.extract("depth", depth1, cmd);
        }

        // flownet
        ncnn::VkMat flow0;
        ncnn::VkMat flow1;
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input0", in0);
            ex.input("input1", in1);
            ex.extract("flow", flow0, cmd);
        }
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input0", in1);
            ex.input("input1", in0);
            ex.extract("flow", flow1, cmd);
        }

        // ctxnet
        ncnn::VkMat ctx0;
        ncnn::VkMat ctx1;
        {
            ncnn::Extractor ex = ctxnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input", in0);
            ex.extract("ctx", ctx0, cmd);
        }
        {
            ncnn::Extractor ex = ctxnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input", in1);
            ex.extract("ctx", ctx1, cmd);
        }

        // interpolation
        ncnn::Mat flow0_w(1);
        ncnn::Mat flow1_w(1);
        flow0_w[0] = 0.5f;
        flow1_w[0] = 0.5f;

        ncnn::VkMat out_padded_gpu;
        {
            ncnn::Extractor ex = interpolation.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input0", in0);
            ex.input("input1", in1);
            ex.input("depth0", depth0);
            ex.input("depth1", depth1);
            ex.input("flow0", flow0);
            ex.input("flow1", flow1);
            ex.input("flow0_w", flow0_w);
            ex.input("flow1_w", flow1_w);
            ex.input("ctx0", ctx0);
            ex.input("ctx1", ctx1);
            ex.extract("output_rectified", out_padded_gpu, cmd);
        }

        // download
        ncnn::Mat out_padded;
        {
            cmd.record_download(out_padded_gpu, out_padded, opt);
        }

        cmd.submit_and_wait();

        // TODO +0.5 before clip
        const float denorm_vals[3] = {255.f, 255.f, 255.f};
        out_padded.substract_mean_normalize(0, denorm_vals);

        ncnn::Mat out;
        ncnn::copy_cut_border(out_padded, out, h_padded/2, h_padded - h_padded/2, w_padded/2, w_padded - w_padded/2);

        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_BGR2RGB);
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
