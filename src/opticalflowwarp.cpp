// dain implemented with ncnn library

#include "dain_ops.h"

int OpticalFlowWarp::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    using namespace ncnn;

    const Mat& image_blob = bottom_blobs[0];
    const Mat& flow_blob = bottom_blobs[1];

    int w = image_blob.w;
    int h = image_blob.h;
    int channels = image_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

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

                // bilinear interpolate
                float v;
                {
                    int x0 = floor(sample_x);
                    int y0 = floor(sample_y);

                    if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1)
                    {
                        v = 0.f;
                    }
                    else
                    {
                        float alpha = sample_x - x0;
                        float beta = sample_y - y0;

                        float v0 = image.row(y0)[x0];
                        float v1 = image.row(y0)[x0 + 1];
                        float v2 = image.row(y0 + 1)[x0];
                        float v3 = image.row(y0 + 1)[x0 + 1];

                        float v4 = v0 * (1 - alpha) + v1 * alpha;
                        float v5 = v2 * (1 - alpha) + v3 * alpha;

                        v = v4 * (1 - beta) + v5 * beta;
                    }
                }

                outptr[0] = v;

                outptr += 1;

                fxptr += 1;
                fyptr += 1;
            }
        }
    }

    return 0;
}
