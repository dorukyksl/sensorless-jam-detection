/* Replays the golden vectors from scripts/export_vectors.py through the
 * float32 C code and compares with the float64 Python reference. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jd_test.h"
#include "jd_detector.h"
#include "jd_emf.h"
#include "jd_math.h"
#include "jd_mras.h"

#define MAX_ROWS 4096
#define MAX_KEYS 32

/* float32 against float64 over about 2000 closed-loop samples */
#define TH_TOL 1e-4   /* rad, electrical */
#define W_TOL  0.1    /* rad/s, electrical (rated is 6231) */

typedef struct {
    char key[32];
    double val;
} kv_t;

typedef struct {
    kv_t kv[MAX_KEYS];
    int nkv;
    char observer[16];
    int n;
    float in[MAX_ROWS][4];
    double th_ref[MAX_ROWS];
    double w_ref[MAX_ROWS];
    int fired_ref[MAX_ROWS];
} vec_t;

static vec_t v;

static double get(const char *key)
{
    for (int i = 0; i < v.nkv; i++) {
        if (strcmp(v.kv[i].key, key) == 0) {
            return v.kv[i].val;
        }
    }
    CHECK(0, "missing key %s", key);
    return 0.0;
}

static int load(const char *name)
{
    char path[512];
    char line[512];
    snprintf(path, sizeof path, "%s/%s", JD_VECTOR_DIR, name);
    FILE *f = fopen(path, "r");
    CHECK(f != NULL, "cannot open %s", path);
    if (f == NULL) {
        return 0;
    }
    memset(&v, 0, sizeof v);
    while (fgets(line, sizeof line, f) != NULL) {
        if (line[0] == '#') {
            char *eq = strchr(line, '=');
            if (eq == NULL) {
                continue;
            }
            *eq = '\0';
            const char *key = line + 2;
            if (strcmp(key, "observer") == 0) {
                sscanf(eq + 1, "%15s", v.observer);
            } else if (strcmp(key, "scenario") != 0 && v.nkv < MAX_KEYS &&
                       strlen(key) < sizeof v.kv[0].key) {
                memcpy(v.kv[v.nkv].key, key, strlen(key) + 1u);
                v.kv[v.nkv].val = strtod(eq + 1, NULL);
                v.nkv++;
            }
            continue;
        }
        double a, b, c, d, th, w;
        int fr;
        if (v.n < MAX_ROWS &&
            sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%d", &a, &b, &c, &d, &th, &w, &fr) == 7) {
            v.in[v.n][0] = (float)a;
            v.in[v.n][1] = (float)b;
            v.in[v.n][2] = (float)c;
            v.in[v.n][3] = (float)d;
            v.th_ref[v.n] = th;
            v.w_ref[v.n] = w;
            v.fired_ref[v.n] = fr;
            v.n++;
        }
    }
    fclose(f);
    CHECK(v.n > 100, "only %d rows in %s", v.n, name);
    return v.n > 100;
}

/* Off-target set-up: what a config generator would bake into flash. */
static jd_model_t model(void)
{
    jd_model_t m;
    const double Ts = get("Ts");
    const double R = get("R");
    const double L = get("L");
    m.Ts = (float)Ts;
    m.a = (float)(R / L);
    m.alpha = (float)exp(-R / L * Ts);
    m.inv_L = (float)(1.0 / L);
    return m;
}

static jd_det_params_t det_params(void)
{
    jd_det_params_t p;
    p.n = (uint16_t)get("det_n");
    p.inv_nts = (float)(1.0 / (get("det_n") * get("Ts")));
    p.a_th = (float)get("det_ath");
    p.w_enable = (float)get("det_wen");
    return p;
}

static int first_fired(const int *f, int n)
{
    for (int k = 0; k < n; k++) {
        if (f[k] != 0) {
            return k;
        }
    }
    return -1;
}

static void run_file(const char *name)
{
    static float th[MAX_ROWS];
    static float w[MAX_ROWS];
    static int fired_c[MAX_ROWS];
    static int fired_py_in[MAX_ROWS];
    if (!load(name)) {
        return;
    }
    const float th0 = (float)get("th0");
    const float w0 = (float)get("w0");
    const float iq0 = (float)get("iq0");
    const float p = (float)get("p");

    if (strcmp(v.observer, "mras") == 0) {
        jd_mras_params_t mp;
        jd_mras_t st;
        mp.model = model();
        mp.psi = (float)get("psi");
        mp.psi_over_L = (float)(get("psi") / get("L"));
        mp.kp = (float)get("op4");
        mp.ki_Ts = (float)(get("op5") * get("Ts"));
        jd_mras_init(&st, th0, w0, 0.0f, iq0);
        for (int k = 0; k < v.n; k++) {
            jd_mras_step(&st, &mp, v.in[k][0], v.in[k][1], v.in[k][2], v.in[k][3]);
            th[k] = st.th;
            w[k] = st.w;
        }
    } else {
        jd_emf_params_t ep;
        jd_emf_t st;
        ep.model = model();
        ep.l1 = (float)get("op4");
        ep.l2 = (float)get("op5");
        ep.kp = (float)get("op6");
        ep.ki_Ts = (float)(get("op7") * get("Ts"));
        ep.emin = (float)get("op8");
        jd_emf_init(&st, th0, w0, 0.0f, iq0, (float)get("psi"));
        for (int k = 0; k < v.n; k++) {
            jd_emf_step(&st, &ep, v.in[k][0], v.in[k][1], v.in[k][2], v.in[k][3]);
            th[k] = st.th;
            w[k] = st.wi;
        }
    }

    double max_th = 0.0;
    double max_w = 0.0;
    for (int k = 0; k < v.n; k++) {
        double d = (double)th[k] - v.th_ref[k];
        d = fmod(d + 3.0 * M_PI, 2.0 * M_PI) - M_PI;
        max_th = fmax(max_th, fabs(d));
        max_w = fmax(max_w, fabs((double)w[k] - v.w_ref[k]));
    }
    printf("  %-14s max |dth| %.2e rad, max |dw| %.3f rad/s\n", name, max_th, max_w);
    CHECK(max_th < TH_TOL, "angle differs from Python by %.3e rad", max_th);
    CHECK(max_w < W_TOL, "speed differs from Python by %.3f rad/s", max_w);

    /* detector fed with the Python speed: must fire on the same sample */
    const jd_det_params_t dp = det_params();
    jd_det_t d;
    jd_det_init(&d, (float)(v.w_ref[0] / (double)p));
    for (int k = 0; k < v.n; k++) {
        fired_py_in[k] = jd_det_step(&d, &dp, (float)(v.w_ref[k] / (double)p));
    }
    const int k_ref = first_fired(v.fired_ref, v.n);
    const int k_py = first_fired(fired_py_in, v.n);
    CHECK(k_ref == k_py, "detector on Python speed fired at %d, Python at %d", k_py, k_ref);

    /* detector fed with the C speed: within one sample */
    jd_det_init(&d, w[0] / p);
    for (int k = 0; k < v.n; k++) {
        fired_c[k] = jd_det_step(&d, &dp, w[k] / p);
    }
    const int k_c = first_fired(fired_c, v.n);
    printf("  %-14s detector fires at sample: Python %d, C %d\n", name, k_ref, k_c);
    if (k_ref < 0) {
        CHECK(k_c == -1, "C detector fired at %d, Python never", k_c);
    } else {
        CHECK(abs(k_c - k_ref) <= 1, "C detector fired at %d, Python at %d", k_c, k_ref);
    }
}

static void test_sincos_accuracy(void)
{
    double max_err = 0.0;
    for (int i = -40000; i <= 40000; i++) {
        const float x = (float)i * 1e-4f;
        float s;
        float c;
        jd_sincos(x, &s, &c);
        max_err = fmax(max_err, fabs((double)s - sin((double)x)));
        max_err = fmax(max_err, fabs((double)c - cos((double)x)));
    }
    printf("  jd_sincos max error on [-4, 4]: %.2e\n", max_err);
    CHECK(max_err < 5e-7, "sincos error %.3e", max_err);
}

static void test_wrap(void)
{
    CHECK(fabsf(jd_wrap(0.5f + JD_TWO_PI) - 0.5f) < 1e-6f, "wrap up");
    CHECK(fabsf(jd_wrap(-0.5f - JD_TWO_PI) + 0.5f) < 1e-6f, "wrap down");
    CHECK(jd_wrap(JD_PI) < JD_PI, "wrap pi");
}

static void test_propagate_zero_speed(void)
{
    /* w = 0: z' = alpha z + (1 - alpha)/a * u/L */
    jd_model_t m = { 1e-4f, 400.0f, expf(-400.0f * 1e-4f), 1.0f / 20e-6f };
    float zr;
    float zi;
    jd_propagate(&m, 10.0f, -5.0f, 1.0f, 2.0f, 0.0f, &zr, &zi);
    const float g = (1.0f - m.alpha) / m.a * m.inv_L;
    CHECK(fabsf(zr - (m.alpha * 10.0f + g * 1.0f)) < 1e-3f, "zr %f", (double)zr);
    CHECK(fabsf(zi - (m.alpha * -5.0f + g * 2.0f)) < 1e-3f, "zi %f", (double)zi);
}

static void test_detector_window(void)
{
    /* constant deceleration of 1000 rad/s^2 at Ts = 1 ms, window 4 */
    jd_det_params_t p = { 4u, 1.0f / (4.0f * 1e-3f), 900.0f, 10.0f };
    jd_det_t d;
    jd_det_init(&d, 500.0f);
    int k_fire = -1;
    for (int k = 0; k < 20; k++) {
        const float w = 500.0f - ((k >= 10) ? (float)(k - 9) : 0.0f);
        if (jd_det_step(&d, &p, w) == 1 && k_fire < 0) {
            k_fire = k;
        }
    }
    /* the window holds 4 samples of slope 1 rad/s per ms from k = 13 */
    CHECK(k_fire == 13, "fired at %d", k_fire);
}

static void test_mras_jam(void) { run_file("mras_jam.csv"); }
static void test_mras_step(void) { run_file("mras_step.csv"); }
static void test_emf_jam(void) { run_file("emf_jam.csv"); }
static void test_emf_step(void) { run_file("emf_step.csv"); }

int main(void)
{
    RUN(test_sincos_accuracy);
    RUN(test_wrap);
    RUN(test_propagate_zero_speed);
    RUN(test_detector_window);
    RUN(test_mras_jam);
    RUN(test_mras_step);
    RUN(test_emf_jam);
    RUN(test_emf_step);
    return TEST_EXIT();
}
