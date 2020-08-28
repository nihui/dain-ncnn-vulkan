// dain implemented with ncnn library

#include "dain_ops.h"

#include "layer_type.h"

#include "correlation.comp.hex.h"
#include "correlation_pack4to1.comp.hex.h"

using namespace ncnn;

Correlation::Correlation()
{
    support_vulkan = true;

    padding = 0;
    pipeline_correlation = 0;
    pipeline_correlation_pack4to1 = 0;
}

int Correlation::create_pipeline(const Option& opt)
{
    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 4);
        pd.set(1, 4);
        pd.set(2, 4);
        pd.set(3, 4);
        pd.set(4, 0);
        pd.set(5, 0.f);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(0 + 0);

    // pack1
    {
        // TODO static
        std::vector<uint32_t> spirv;
        compile_spirv_module(correlation_comp_data, sizeof(correlation_comp_data), opt, spirv);

        pipeline_correlation = new Pipeline(vkdev);
        pipeline_correlation->set_optimal_local_size_xyz();
        pipeline_correlation->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack4to1
    {
        // TODO static
        std::vector<uint32_t> spirv;
        compile_spirv_module(correlation_pack4to1_comp_data, sizeof(correlation_pack4to1_comp_data), opt, spirv);

        pipeline_correlation_pack4to1 = new Pipeline(vkdev);
        pipeline_correlation_pack4to1->set_optimal_local_size_xyz();
        pipeline_correlation_pack4to1->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int Correlation::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_correlation;
    pipeline_correlation = 0;

    delete pipeline_correlation_pack4to1;
    pipeline_correlation_pack4to1 = 0;

    return 0;
}

int Correlation::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    return 0;
}

int Correlation::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& a = bottom_blobs[0];
    const Mat& b = bottom_blobs[1];

    int w = a.w;
    int h = a.h;
    int channels = a.c;

    const int md = 4;
    const int pad = 4;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, 9*9);
    if (top_blob.empty())
        return -100;

    Mat a_bordered;
    Mat b_bordered;
    copy_make_border(a, a_bordered, pad, pad, pad, pad, BORDER_CONSTANT, 0.f);
    copy_make_border(b, b_bordered, pad, pad, pad, pad, BORDER_CONSTANT, 0.f);
    if (a_bordered.empty() || b_bordered.empty())
        return -100;

    #pragma omp parallel for
    for (int tj = 0; tj < 9; tj++)
    {
        for (int ti = 0; ti < 9; ti++)
        {
            int tindex = tj * 9 + ti;

            Mat outm = top_blob.channel(tindex);
            outm.fill(0.f);

            for (int q = 0; q < channels; q++)
            {
                const Mat am = a_bordered.channel(q);
                const Mat bm = b_bordered.channel(q);

                for (int y = 0; y < h; y++)
                {
                    int y1 = y + 4;
                    int y2 = y + tj;

                    const float* aptr = am.row(y1);
                    const float* bptr = bm.row(y2);

                    float* outptr = outm.row(y);

                    for (int x = 0; x < w; x++)
                    {
                        int x1 = x + 4;
                        int x2 = x + ti;

                        float va = aptr[x1];
                        float vb = bptr[x2];

                        outptr[x] += va * vb;
                    }
                }
            }


            for (int y = 0; y < h; y++)
            {
                float* outptr = outm.row(y);

                for (int x = 0; x < w; x++)
                {
                    outptr[x] /= channels;
                }
            }
        }
    }

    return 0;
}

int Correlation::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& a = bottom_blobs[0];
    const VkMat& b = bottom_blobs[1];

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int elempack = a.elempack;

//     const int md = 4;
//     const int pad = 4;

    VkMat a_bordered;
    VkMat b_bordered;
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(a, a_bordered, cmd, opt_pad);
        padding->forward(b, b_bordered, cmd, opt_pad);
        if (a_bordered.empty() || b_bordered.empty())
            return -100;
    }

    size_t out_elemsize = opt.use_fp16_storage ? 2u : 4u;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(w, h, 9*9, out_elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = a_bordered;
    bindings[1] = b_bordered;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(8);
    constants[0].i = a_bordered.w;
    constants[1].i = a_bordered.h;
    constants[2].i = a_bordered.c;
    constants[3].i = a_bordered.cstep;
    constants[4].i = top_blob.w;
    constants[5].i = top_blob.h;
    constants[6].i = top_blob.c;
    constants[7].i = top_blob.cstep;

    if (elempack == 4)
    {
        cmd.record_pipeline(pipeline_correlation_pack4to1, bindings, constants, top_blob);
    }
    else // if (elempack == 1)
    {
        cmd.record_pipeline(pipeline_correlation, bindings, constants, top_blob);
    }

    return 0;
}
