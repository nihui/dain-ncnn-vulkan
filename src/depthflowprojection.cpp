// dain implemented with ncnn library

#include "dain_ops.h"

int DepthFlowProjection::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    using namespace ncnn;

    const Mat& flow_blob = bottom_blobs[0];
    const Mat& depth_blob = bottom_blobs[1];

    int w = flow_blob.w;
    int h = flow_blob.h;
    int channels = flow_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    top_blob.fill(0.f);

    Mat fxydc(w, h);
    fxydc.fill(0.f);

    // projection
    {
        Mat fxdm = top_blob.channel(0);
        Mat fydm = top_blob.channel(1);

        const float* fxptr = flow_blob.channel(0);
        const float* fyptr = flow_blob.channel(1);
        const float* dptr = depth_blob.channel(0);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float flow_x = fxptr[0];
                float flow_y = fyptr[0];
                float depth = dptr[0];

                float sample_x = x + flow_x;
                float sample_y = y + flow_y;

                // 2x2
                {
                    int x0 = floor(sample_x);
                    int y0 = floor(sample_y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1)
                    {
                        // discard out of image
                    }
                    else
                    {
                        fxdm.row(y0)[x0] -= flow_x * depth;
                        fxdm.row(y0)[x1] -= flow_x * depth;
                        fxdm.row(y1)[x0] -= flow_x * depth;
                        fxdm.row(y1)[x1] -= flow_x * depth;

                        fydm.row(y0)[x0] -= flow_y * depth;
                        fydm.row(y0)[x1] -= flow_y * depth;
                        fydm.row(y1)[x0] -= flow_y * depth;
                        fydm.row(y1)[x1] -= flow_y * depth;

                        fxydc.row(y0)[x0] += depth;
                        fxydc.row(y0)[x1] += depth;
                        fxydc.row(y1)[x0] += depth;
                        fxydc.row(y1)[x1] += depth;
                    }
                }


                fxptr += 1;
                fyptr += 1;
                dptr += 1;
            }
        }
    }

    // average
    {
        float* fxdptr = top_blob.channel(0);
        float* fydptr = top_blob.channel(1);
        const float* fxydcptr = fxydc;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float count = fxydcptr[0];
                if (count > 0.f)
                {
                    fxdptr[0] /= count;
                    fydptr[0] /= count;
                }

                fxdptr += 1;
                fydptr += 1;
                fxydcptr += 1;
            }
        }
    }

    // fill hole
    {
        float* fxdptr = top_blob.channel(0);
        float* fydptr = top_blob.channel(1);
        const float* fxydcptr = fxydc;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float count = fxydcptr[0];
                if (count <= 0.f)
                {
                    // search along the four directions, 0/90/180/270, until finding at least one

                    // left
                    int left_x = x;
                    float left_count = 0.f;
                    while (left_count <= 0.f && left_x - 1 >= 0)
                    {
                        left_count = fxydc.row(y)[left_x];
                        left_x--;
                    }

                    // right
                    int right_x = x;
                    float right_count = 0.f;
                    while (right_count <= 0.f && right_x + 1 <= w - 1)
                    {
                        right_count = fxydc.row(y)[right_x];
                        right_x++;
                    }

                    // up
                    int up_y = y;
                    float up_count = 0.f;
                    while (up_count <= 0.f && up_y - 1 >= 0)
                    {
                        up_count = fxydc.row(up_y)[x];
                        up_y--;
                    }

                    // down
                    int down_y = y;
                    float down_count = 0.f;
                    while (down_count <= 0.f && down_y + 1 <= h - 1)
                    {
                        down_count = fxydc.row(down_y)[x];
                        down_y++;
                    }

                    if (left_count + right_count + up_count + down_count > 0.f)
                    {
                        float fxd = 0.f;
                        float fyd = 0.f;
                        float new_count = 0.f;
                        if (left_count > 0.f)
                        {
                            fxd += top_blob.channel(0).row(y)[left_x];
                            fyd += top_blob.channel(1).row(y)[left_x];
                            new_count += 1;
                        }
                        if (right_count > 0.f)
                        {
                            fxd += top_blob.channel(0).row(y)[right_x];
                            fyd += top_blob.channel(1).row(y)[right_x];
                            new_count += 1;
                        }
                        if (up_count > 0.f)
                        {
                            fxd += top_blob.channel(0).row(up_y)[x];
                            fyd += top_blob.channel(1).row(up_y)[x];
                            new_count += 1;
                        }
                        if (down_count > 0.f)
                        {
                            fxd += top_blob.channel(0).row(down_y)[x];
                            fyd += top_blob.channel(1).row(down_y)[x];
                            new_count += 1;
                        }

                        fxdptr[0] = fxd / new_count;
                        fydptr[0] = fyd / new_count;
                    }
                }

                fxdptr += 1;
                fydptr += 1;
                fxydcptr += 1;
            }
        }
    }

    return 0;
}
