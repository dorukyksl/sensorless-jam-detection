/* Back EMF observer in the estimated frame plus a type 2 PLL.
 * Mirrors emf_step() in src/jamsim/observers.py. */
#ifndef JD_EMF_H
#define JD_EMF_H

#include "jd_math.h"

typedef struct {
    jd_model_t model;
    float l1;      /* current correction gain [-] */
    float l2;      /* EMF correction gain [V/A] */
    float kp;      /* PLL gains */
    float ki_Ts;
    float emin;    /* floor on |e_hat| when normalising [V] */
} jd_emf_params_t;

typedef struct {
    float ig;   /* current estimate, estimated frame [A] */
    float id;
    float eg;   /* back EMF estimate [V] */
    float ed;
    float th;   /* electrical angle estimate [rad] */
    float wf;   /* PLL output, turns the frame [rad/s] */
    float wi;   /* PLL integrator, speed estimate [rad/s] */
} jd_emf_t;

void jd_emf_init(jd_emf_t *st, float th, float w, float ig, float id, float psi);

/* Same interface as jd_mras_step(). The speed estimate is st->wi. */
void jd_emf_step(jd_emf_t *st, const jd_emf_params_t *p,
                 float ia, float ib, float ua, float ub);

#endif
