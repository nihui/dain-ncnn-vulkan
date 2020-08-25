// dain implemented with ncnn library

#include "dain_ops.h"

int Correlation::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    using namespace ncnn;

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
