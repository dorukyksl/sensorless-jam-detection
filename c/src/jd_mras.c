#include "jd_mras.h"

void jd_mras_init(jd_mras_t *st, float th, float w, float id, float iq)
{
    st->id = id;
    st->iq = iq;
    st->w = w;
    st->integ = w;
    st->th = th;
}

void jd_mras_step(jd_mras_t *st, const jd_mras_params_t *p,
                  float ia, float ib, float ua, float ub)
{
    const float Ts = p->model.Ts;
    float w = st->w;
    float th = st->th;
    float ug;
    float ud;
    float zr;
    float zi;
    float ig;
    float idl;

    /* voltage in the estimated frame at the middle of the last period */
    jd_park(ua, ub, th + 0.5f * w * Ts, &ug, &ud);
    /* adjustable model, input v - j psi w */
    jd_propagate(&p->model, st->id, st->iq, ug, ud - p->psi * w, w, &zr, &zi);
    th = jd_wrap(th + w * Ts);
    jd_park(ia, ib, th, &ig, &idl);
    /* Popov adaptation signal, Westin eq. (4.8) */
    const float eps = (zi - idl) * (ig + p->psi_over_L) - (zr - ig) * idl;
    st->integ += p->ki_Ts * eps;
    w = p->kp * eps + st->integ;

    st->id = zr;
    st->iq = zi;
    st->w = w;
    st->th = th;
}
