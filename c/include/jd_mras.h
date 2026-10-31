/* MRAS speed and angle observer (Westin 2016, eq. 4.7 to 4.9).
 * Mirrors mras_step() in src/jamsim/observers.py. */
#ifndef JD_MRAS_H
#define JD_MRAS_H

#include "jd_math.h"

typedef struct {
    jd_model_t model;
    float psi;         /* PM flux [Vs/rad] */
    float psi_over_L;  /* [A] */
    float kp;          /* adaptation PI gains */
    float ki_Ts;       /* ki * Ts */
} jd_mras_params_t;

typedef struct {
    float id;     /* adjustable model current, estimated frame [A] */
    float iq;
    float w;      /* electrical speed estimate [rad/s] */
    float integ;  /* PI integrator [rad/s] */
    float th;     /* electrical angle estimate [rad] */
} jd_mras_t;

void jd_mras_init(jd_mras_t *st, float th, float w, float id, float iq);

/* ia, ib: currents sampled now. ua, ub: voltage applied over the period
 * that just ended. Updates st->th and st->w for use in this period. */
void jd_mras_step(jd_mras_t *st, const jd_mras_params_t *p,
                  float ia, float ib, float ua, float ub);

#endif
