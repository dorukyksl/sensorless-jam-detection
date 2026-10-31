/* Jam / snag detector: mean deceleration over the last n samples above a
 * threshold while the speed is above an enable level. Latches once fired.
 * Mirrors det_step() in src/jamsim/detector.py. */
#ifndef JD_DETECTOR_H
#define JD_DETECTOR_H

#include <stdint.h>

#define JD_DET_NMAX 256u

typedef struct {
    uint16_t n;        /* window length in samples, 1..JD_DET_NMAX-1 */
    float inv_nts;     /* 1 / (n Ts) */
    float a_th;        /* deceleration threshold [rad/s^2], mechanical */
    float w_enable;    /* enable speed [rad/s], mechanical */
} jd_det_params_t;

typedef struct {
    float buf[JD_DET_NMAX];
    uint16_t head;
    uint16_t seen;     /* saturates at JD_DET_NMAX */
    uint8_t fired;
} jd_det_t;

void jd_det_init(jd_det_t *d, float w0);

/* Push the newest mechanical speed estimate. Returns 1 when fired. */
int jd_det_step(jd_det_t *d, const jd_det_params_t *p, float w);

#endif
