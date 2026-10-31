/* Small float32 helpers for the per-sample path. No libm, no doubles. */
#ifndef JD_MATH_H
#define JD_MATH_H

#define JD_PI     3.14159265358979f
#define JD_TWO_PI 6.28318530717959f

/* Wrap an angle to [-pi, pi). */
float jd_wrap(float x);

/* sin and cos of x, |x| up to a few pi. Max error about 1e-7 on [-pi, pi]. */
void jd_sincos(float x, float *s, float *c);

/* Rotate alpha-beta into a frame at angle th. */
static inline void jd_park(float xa, float xb, float th, float *xg, float *xd)
{
    float s;
    float c;
    jd_sincos(th, &s, &c);
    *xg = xa * c + xb * s;
    *xd = -xa * s + xb * c;
}

/* Stator current model shared by both observers. */
typedef struct {
    float Ts;     /* control period [s] */
    float a;      /* R / L [1/s] */
    float alpha;  /* exp(-a Ts), computed off-target */
    float inv_L;  /* 1 / L [1/H] */
} jd_model_t;

/* Exact step over Ts of dz/dt = (-a - j w) z + u / L with w and u held.
 * z = zr + j zi is the current in a frame turning at w. */
void jd_propagate(const jd_model_t *m, float zr, float zi, float ur, float ui,
                  float w, float *zr_out, float *zi_out);

#endif
