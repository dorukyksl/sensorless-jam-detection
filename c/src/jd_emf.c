#include "jd_emf.h"

void jd_emf_init(jd_emf_t *st, float th, float w, float ig, float id, float psi)
{
    st->ig = ig;
    st->id = id;
    st->eg = 0.0f;
    st->ed = psi * w;
    st->th = th;
    st->wf = w;
    st->wi = w;
}

void jd_emf_step(jd_emf_t *st, const jd_emf_params_t *p,
                 float ia, float ib, float ua, float ub)
{
    const float Ts = p->model.Ts;
    const float wf = st->wf;
    float th = st->th;
    float ug;
    float ud;
    float zr;
    float zi;
    float ig;
    float idl;

    jd_park(ua, ub, th + 0.5f * wf * Ts, &ug, &ud);
    jd_propagate(&p->model, st->ig, st->id, ug - st->eg, ud - st->ed, wf, &zr, &zi);
    th = jd_wrap(th + wf * Ts);
    jd_park(ia, ib, th, &ig, &idl);
    const float rg = ig - zr;
    const float rd = idl - zi;
    st->ig = zr + p->l1 * rg;
    st->id = zi + p->l1 * rd;
    const float eg = st->eg - p->l2 * rg;
    const float ed = st->ed - p->l2 * rd;
    st->eg = eg;
    st->ed = ed;

    /* e_hat = |e| (-sin d, cos d), d = theta - theta_hat */
    float mag = __builtin_sqrtf(eg * eg + ed * ed);
    if (mag < p->emin) {
        mag = p->emin;
    }
    float err = -eg / mag;
    if (ed < 0.0f) {
        err = -err;
    }
    st->wi += p->ki_Ts * err;
    st->wf = p->kp * err + st->wi;
    st->th = th;
}
