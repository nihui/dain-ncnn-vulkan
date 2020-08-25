// dain implemented with ncnn library

#include "dain_ops.h"

int FilterInterpolation::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    using namespace ncnn;

    const Mat& image_blob = bottom_blobs[0];
    const Mat& flow_blob = bottom_blobs[1];
    const Mat& filter_blob = bottom_blobs[2];

    int w = image_blob.w;
    int h = image_blob.h;
    int channels = image_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    const int filter_size = 4; // (int)sqrt(filter_blob.c);

    #pragma omp parallel for
    for (int q = 0; q < channels; q++)
    {
        float* outptr = top_blob.channel(q);

        const Mat image = image_blob.channel(q);
        const float* fxptr = flow_blob.channel(0);
        const float* fyptr = flow_blob.channel(1);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float flow_x = fxptr[0];
                float flow_y = fyptr[0];

                float sample_x = x + flow_x;
                float sample_y = y + flow_y;

                if (sample_x < 0.f || sample_y < 0.f || sample_x >= w - 1 || sample_y >= h - 1 || fabs(flow_x) > w / 2.f || fabs(flow_y) > h / 2.f)
                {
                    // the warping data is out of range, we fill it with zeros
                    outptr[0] = image.row(y)[x];
                }
                else
                {
                    // 4x4
                    int x1 = floor(sample_x);
                    int y1 = floor(sample_y);
                    int x0 = x1 - 1;
                    int y0 = y1 - 1;
                    int x2 = x1 + 1;
                    int y2 = y1 + 1;
                    int x3 = x1 + 2;
                    int y3 = y1 + 2;

                    float alpha = sample_x - x1;
                    float beta = sample_y - y1;

                    // sanitize out of image
                    x0 = std::min(std::max(x0, 0), w - 1);
                    x1 = std::min(std::max(x1, 0), w - 1);
                    x2 = std::min(std::max(x2, 0), w - 1);
                    x3 = std::min(std::max(x3, 0), w - 1);
                    y0 = std::min(std::max(y0, 0), h - 1);
                    y1 = std::min(std::max(y1, 0), h - 1);
                    y2 = std::min(std::max(y2, 0), h - 1);
                    y3 = std::min(std::max(y3, 0), h - 1);

                    float v00 = image.row(y0)[x0];
                    float v01 = image.row(y0)[x1];
                    float v02 = image.row(y0)[x2];
                    float v03 = image.row(y0)[x3];

                    float v10 = image.row(y1)[x0];
                    float v11 = image.row(y1)[x1];
                    float v12 = image.row(y1)[x2];
                    float v13 = image.row(y1)[x3];

                    float v20 = image.row(y2)[x0];
                    float v21 = image.row(y2)[x1];
                    float v22 = image.row(y2)[x2];
                    float v23 = image.row(y2)[x3];

                    float v30 = image.row(y3)[x0];
                    float v31 = image.row(y3)[x1];
                    float v32 = image.row(y3)[x2];
                    float v33 = image.row(y3)[x3];

                    float w00 = filter_blob.channel(0).row(y)[x];
                    float w01 = filter_blob.channel(1).row(y)[x];
                    float w02 = filter_blob.channel(2).row(y)[x];
                    float w03 = filter_blob.channel(3).row(y)[x];
                    float w10 = filter_blob.channel(4).row(y)[x];
                    float w11 = filter_blob.channel(5).row(y)[x];
                    float w12 = filter_blob.channel(6).row(y)[x];
                    float w13 = filter_blob.channel(7).row(y)[x];
                    float w20 = filter_blob.channel(8).row(y)[x];
                    float w21 = filter_blob.channel(9).row(y)[x];
                    float w22 = filter_blob.channel(10).row(y)[x];
                    float w23 = filter_blob.channel(11).row(y)[x];
                    float w30 = filter_blob.channel(12).row(y)[x];
                    float w31 = filter_blob.channel(13).row(y)[x];
                    float w32 = filter_blob.channel(14).row(y)[x];
                    float w33 = filter_blob.channel(15).row(y)[x];

                    float TL = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                    float TR = v02 * w02 + v03 * w03 + v12 * w12 + v13 * w13;
                    float BL = v20 * w20 + v21 * w21 + v30 * w30 + v31 * w31;
                    float BR = v22 * w22 + v23 * w23 + v32 * w32 + v33 * w33;

                    float T = TL * (1 - alpha) + TR * alpha;
                    float B = BL * (1 - alpha) + BR * alpha;
                    float v = T * (1 - beta) + B * beta;

                    outptr[0] = v;
                }


                fxptr += 1;
                fyptr += 1;
                outptr += 1;
            }
        }
    }

    return 0;
}
