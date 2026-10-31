#include "jd_detector.h"

void jd_det_init(jd_det_t *d, float w0)
{
    for (uint16_t i = 0u; i < JD_DET_NMAX; i++) {
        d->buf[i] = w0;
    }
    d->head = 0u;
    d->seen = 0u;
    d->fired = 0u;
}

int jd_det_step(jd_det_t *d, const jd_det_params_t *p, float w)
{
    const uint16_t head = d->head;
    const uint16_t old_idx = (uint16_t)((head + JD_DET_NMAX - p->n) % JD_DET_NMAX);
    const float old = (d->seen >= p->n) ? d->buf[old_idx] : w;
    const int full = (d->seen >= p->n);

    d->buf[head] = w;
    d->head = (uint16_t)((head + 1u) % JD_DET_NMAX);
    if (d->seen < JD_DET_NMAX) {
        d->seen++;
    }
    if (d->fired != 0u) {
        return 1;
    }
    if (full && (w > p->w_enable) && ((old - w) * p->inv_nts > p->a_th)) {
        d->fired = 1u;
        return 1;
    }
    return 0;
}
