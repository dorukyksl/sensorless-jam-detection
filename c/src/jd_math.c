#include <stdint.h>

#include "jd_math.h"

float jd_wrap(float x)
{
    while (x >= JD_PI) {
        x -= JD_TWO_PI;
    }
    while (x < -JD_PI) {
        x += JD_TWO_PI;
    }
    return x;
}

void jd_sincos(float x, float *s, float *c)
{
    /* nearest quadrant, then Cody-Waite reduction with pi/2 split in two */
    const float fq = x * 0.636619772f;
    const int32_t q = (int32_t)(fq + ((fq >= 0.0f) ? 0.5f : -0.5f));
    const float qf = (float)q;
    const float r = (x - qf * 1.57079637050628662f) + qf * 4.37113900018624e-8f;
    const float r2 = r * r;
    /* Taylor to r^9 and r^10, |r| <= pi/4 */
    const float sr = r * (1.0f + r2 * (-1.66666667e-1f + r2 * (8.33333333e-3f
                     + r2 * (-1.98412698e-4f + r2 * 2.75573192e-6f))));
    const float cr = 1.0f + r2 * (-0.5f + r2 * (4.16666667e-2f
                     + r2 * (-1.38888889e-3f + r2 * 2.48015873e-5f)));
    switch ((uint32_t)q & 3u) {
    case 0u:
        *s = sr;
        *c = cr;
        break;
    case 1u:
        *s = cr;
        *c = -sr;
        break;
    case 2u:
        *s = -sr;
        *c = -cr;
        break;
    default:
        *s = -cr;
        *c = sr;
        break;
    }
}

void jd_propagate(const jd_model_t *m, float zr, float zi, float ur, float ui,
                  float w, float *zr_out, float *zi_out)
{
    float s;
    float c;
    jd_sincos(w * m->Ts, &s, &c);
    const float er = m->alpha * c;
    const float ei = -m->alpha * s;
    const float nr = er - 1.0f;
    const float ni = ei;
    const float inv_den = 1.0f / (m->a * m->a + w * w);
    const float gr = (-m->a * nr - w * ni) * inv_den;
    const float gi = (w * nr - m->a * ni) * inv_den;
    ur *= m->inv_L;
    ui *= m->inv_L;
    *zr_out = er * zr - ei * zi + gr * ur - gi * ui;
    *zi_out = er * zi + ei * zr + gr * ui + gi * ur;
}
