// dain implemented with ncnn library

#include "dain.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "dain_preproc.comp.hex.h"
#include "dain_postproc.comp.hex.h"

#include "dain_ops.h"

DEFINE_LAYER_CREATOR(Correlation)
DEFINE_LAYER_CREATOR(OpticalFlowWarp)
DEFINE_LAYER_CREATOR(DepthFlowProjection)
DEFINE_LAYER_CREATOR(FilterInterpolation)

DAIN::DAIN()
{
    tilesize = 256;
    prepadding = 32;

    vkdev = ncnn::get_gpu_device(0);
    dain_preproc = 0;
    dain_postproc = 0;
}

DAIN::~DAIN()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete dain_preproc;
        delete dain_postproc;
    }
}

int DAIN::load()
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;

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

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(dain_preproc_comp_data, sizeof(dain_preproc_comp_data), opt, spirv);
                }
            }

            dain_preproc = new ncnn::Pipeline(vkdev);
            dain_preproc->set_optimal_local_size_xyz(8, 8, 3);
            dain_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(dain_postproc_comp_data, sizeof(dain_postproc_comp_data), opt, spirv);
                }
            }

            dain_postproc = new ncnn::Pipeline(vkdev);
            dain_postproc->set_optimal_local_size_xyz(8, 8, 3);
            dain_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int DAIN::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, ncnn::Mat& outimage) const
{
    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = depthnet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    // each tile 100x100
    const int xtiles = (w_padded + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h_padded + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    fprintf(stderr, "tiles %d %d\n", xtiles, ytiles);

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

//         fprintf(stderr, "in_tile_y0 %d %d\n", in_tile_y0, in_tile_y1);

        ncnn::Mat in0;
        ncnn::Mat in1;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            in0 = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixel0data + in_tile_y0 * w * channels, (size_t)channels, 1);
            in1 = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixel1data + in_tile_y0 * w * channels, (size_t)channels, 1);
        }
        else
        {
#if _WIN32
            in0 = ncnn::Mat::from_pixels(pixel0data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR, w, (in_tile_y1 - in_tile_y0));
            in1 = ncnn::Mat::from_pixels(pixel1data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR, w, (in_tile_y1 - in_tile_y0));
#else
            in0 = ncnn::Mat::from_pixels(pixel0data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB2BGR, w, (in_tile_y1 - in_tile_y0));
            in1 = ncnn::Mat::from_pixels(pixel1data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB2BGR, w, (in_tile_y1 - in_tile_y0));
#endif
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in0_gpu;
        ncnn::VkMat in1_gpu;
        {
            cmd.record_clone(in0, in0_gpu, opt);
            cmd.record_clone(in1, in1_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = yi * TILE_SIZE_Y;
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, (out_tile_y1 - out_tile_y0), (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, (out_tile_y1 - out_tile_y0), channels, (size_t)4u, 1, blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            // preproc
            ncnn::VkMat in0_tile_gpu;
            ncnn::VkMat in1_tile_gpu;
            {
                // crop tile
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w_padded) + prepadding;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h_padded) + prepadding;

                in0_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in0_gpu;
                bindings[1] = in0_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(9);
                constants[0].i = in0_gpu.w;
                constants[1].i = in0_gpu.h;
                constants[2].i = in0_gpu.cstep;
                constants[3].i = in0_tile_gpu.w;
                constants[4].i = in0_tile_gpu.h;
                constants[5].i = in0_tile_gpu.cstep;
                constants[6].i = prepadding;
                constants[7].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                constants[8].i = xi * TILE_SIZE_X;

                cmd.record_pipeline(dain_preproc, bindings, constants, in0_tile_gpu);
            }
            {
                // crop tile
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w_padded) + prepadding;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h_padded) + prepadding;

                in1_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in1_gpu;
                bindings[1] = in1_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(9);
                constants[0].i = in1_gpu.w;
                constants[1].i = in1_gpu.h;
                constants[2].i = in1_gpu.cstep;
                constants[3].i = in1_tile_gpu.w;
                constants[4].i = in1_tile_gpu.h;
                constants[5].i = in1_tile_gpu.cstep;
                constants[6].i = prepadding;
                constants[7].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                constants[8].i = xi * TILE_SIZE_X;

                cmd.record_pipeline(dain_preproc, bindings, constants, in1_tile_gpu);
            }

//             fprintf(stderr, "in0_tile_gpu %d %d\n", in0_tile_gpu.w, in0_tile_gpu.h);

            // depthnet
            ncnn::VkMat depth0;
            ncnn::VkMat depth1;
            {
                ncnn::Extractor ex = depthnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input", in0_tile_gpu);
                ex.extract("depth", depth0, cmd);
            }
            {
                ncnn::Extractor ex = depthnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input", in1_tile_gpu);
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

                ex.input("input0", in0_tile_gpu);
                ex.input("input1", in1_tile_gpu);
                ex.extract("flow", flow0, cmd);
            }
            {
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input0", in1_tile_gpu);
                ex.input("input1", in0_tile_gpu);
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

                ex.input("input", in0_tile_gpu);
                ex.extract("ctx", ctx0, cmd);
            }
            {
                ncnn::Extractor ex = ctxnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input", in1_tile_gpu);
                ex.extract("ctx", ctx1, cmd);
            }

            // interpolation
            ncnn::Mat flow0_w(1);
            ncnn::Mat flow1_w(1);
            flow0_w[0] = 0.5f;
            flow1_w[0] = 0.5f;

            ncnn::VkMat out_gpu_padded;
            {
                ncnn::Extractor ex = interpolation.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input0", in0_tile_gpu);
                ex.input("input1", in1_tile_gpu);
                ex.input("depth0", depth0);
                ex.input("depth1", depth1);
                ex.input("flow0", flow0);
                ex.input("flow1", flow1);
                ex.input("flow0_w", flow0_w);
                ex.input("flow1_w", flow1_w);
                ex.input("ctx0", ctx0);
                ex.input("ctx1", ctx1);
                ex.extract("output_rectified", out_gpu_padded, cmd);
            }

            // postproc
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = out_gpu_padded;
                bindings[1] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(9);
                constants[0].i = out_gpu_padded.w;
                constants[1].i = out_gpu_padded.h;
                constants[2].i = out_gpu_padded.cstep;
                constants[3].i = out_gpu.w;
                constants[4].i = out_gpu.h;
                constants[5].i = out_gpu.cstep;
                constants[6].i = prepadding;
                constants[7].i = prepadding;
                constants[8].i = xi * TILE_SIZE_X;

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
                dispatcher.h = out_gpu.h;
                dispatcher.c = 3;

                cmd.record_pipeline(dain_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }

            fprintf(stderr, "%.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
        }

        // download
        {
            ncnn::Mat out;

            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data + out_tile_y0 * w * channels, (size_t)channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {
#if _WIN32
                out.to_pixels((unsigned char*)outimage.data + out_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR);
#else
                out.to_pixels((unsigned char*)outimage.data + out_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB);
#endif
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
