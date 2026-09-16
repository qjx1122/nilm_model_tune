/* nilm_edge.c — NILM 边缘实时推理引擎（P0）
 *
 * 实现：
 *   A 波形特征层：每包（128 点×6 通道）按相累加 Σu·i / Σu² / Σi² → 6s 桶结束时
 *     换算 P（有功）/Vrms/Irms/S（视在），三相求和（按 bundle power_type 选口径）；
 *   B 流式引擎：6s 功率值进 1024 深度环形缓冲；每模型 window 点齐即触发前向；
 *     空桶用上一桶值填充（等价训练期 ffill 语义），连续空桶 >300（30min）判长缺口并重置暖机；
 *   C 推理核：Transformer encoder（pre-LN）seq2point 纯 C 前向（双精度激活，float32 权重）。
 *
 * 数值口径与训练端严格对齐：
 *   - 窗口取 [center-window/2, center+window/2)（src/data.py build_splits 同语义）；
 *   - 归一化统计量取自 bundle（= 训练 npz train 段，export_edge_bundle.py 导出）；
 *   - pe（位置编码）由导出端提供（与训练 forward 位相同源）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nilm_edge.h"

#define TICK_SEC     6.0
#define RAW_RING_MAX 1024
#define CARRY_LIMIT  300 /* 连续空桶上限（300×6s=30min，同制备期 long-gap 剔除语义） */
#define LN_EPS       1e-5

typedef struct {
    int d, nhead, num_layers, ff, window, power_type, head_dim;
    double x_mean, x_std, y_mean, y_std, threshold;
    /* 权重（float32，布局见 export_edge_bundle.py 文档） */
    float *proj_w, *proj_b, *pe;
    float *ln1_w, *ln1_b, *qkvw, *qkvb, *outw, *outb;
    float *ln2_w, *ln2_b, *l1w, *l1b, *l2w, *l2b;
    float *fn_w, *fn_b, *h1w, *h1b, *h2w, *h2b;
    /* 工作区（双精度激活） */
    double *xs, *xbuf, *qkv, *av, *ffn, *sc;
    /* 结果 FIFO（时间序） */
    NilmEdgeResult rq[NILM_EDGE_RESULT_RING];
    int r_head, r_tail;
} EdgeModel;

static struct {
    int started;
    EdgeModel models[NILM_EDGE_MAX_MODELS];
    int n_models, power_type_set, power_type;
    int64_t cur_bucket;
    int have_bucket;
    double sum_ui[3], sum_u2[3], sum_i2[3];
    int64_t bucket_n;
    double raw_val[RAW_RING_MAX], raw_ts[RAW_RING_MAX];
    int64_t raw_count;
    double last_value;
    int have_last, carry_run;
    char err[256];
} E;

/* ---------------- 基础 ---------------- */

static void seterr(const char *msg)
{
    snprintf(E.err, sizeof(E.err), "%s", msg);
}

const char *nilm_edge_version(void) { return "nilm-edge 0.1.0 (P0, bundle v1)"; }
const char *nilm_edge_last_error(void) { return E.err; }

static void free_model(EdgeModel *M)
{
    free(M->proj_w); free(M->proj_b); free(M->pe);
    free(M->ln1_w); free(M->ln1_b); free(M->qkvw); free(M->qkvb);
    free(M->outw); free(M->outb); free(M->ln2_w); free(M->ln2_b);
    free(M->l1w); free(M->l1b); free(M->l2w); free(M->l2b);
    free(M->fn_w); free(M->fn_b); free(M->h1w); free(M->h1b); free(M->h2w); free(M->h2b);
    free(M->xs); free(M->xbuf); free(M->qkv); free(M->av); free(M->ffn); free(M->sc);
    memset(M, 0, sizeof(*M));
}

static void free_all(void)
{
    for (int i = 0; i < E.n_models; i++)
        free_model(&E.models[i]);
    memset(&E, 0, sizeof(E));
}

int nilm_edge_start(void)
{
    free_all();
    E.started = 1;
    return 0;
}

void nilm_edge_shutdown(void) { free_all(); }

/* ---------------- 部署包加载 ---------------- */

static int rdf(FILE *f, float *dst, size_t n)
{
    if (n == 0) return 0;
    return fread(dst, sizeof(float), n, f) == n ? 0 : -1;
}

static int load_bundle(EdgeModel *M, const char *path)
{
    memset(M, 0, sizeof(*M));
    FILE *f = fopen(path, "rb");
    if (!f) { seterr("bundle 文件打不开"); return -1; }
    char magic[4];
    int32_t ver, hdr[6];
    double st[5];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NEDG", 4) != 0) {
        seterr("bundle magic 不符（应为 NEDG）"); fclose(f); return -1;
    }
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != 1) {
        seterr("bundle 版本不支持（应为 1）"); fclose(f); return -1;
    }
    if (fread(hdr, sizeof(int32_t), 6, f) != 6 || fread(st, sizeof(double), 5, f) != 5) {
        seterr("bundle 头部不完整"); fclose(f); return -1;
    }
    M->d = hdr[0]; M->nhead = hdr[1]; M->num_layers = hdr[2]; M->ff = hdr[3];
    M->window = hdr[4]; M->power_type = hdr[5];
    if (M->d <= 0 || M->nhead <= 0 || M->d % M->nhead != 0 || M->num_layers < 1 ||
        M->ff <= 0 || M->window <= 1 || M->window > RAW_RING_MAX ||
        (M->power_type != 0 && M->power_type != 1)) {
        seterr("bundle 头部参数非法"); fclose(f); return -1;
    }
    M->head_dim = M->d / M->nhead;
    M->x_mean = st[0]; M->x_std = st[1]; M->y_mean = st[2]; M->y_std = st[3]; M->threshold = st[4];

    const int d = M->d, L = M->num_layers, ff = M->ff, W = M->window, half = d / 2;
    int ok = 1;
    #define ALLOC(p, n) do { p = malloc(sizeof(float) * (size_t)(n)); ok = ok && (p || (n) == 0); } while (0)
    ALLOC(M->proj_w, d);            ALLOC(M->proj_b, d);
    ALLOC(M->pe, (size_t)W * d);
    ALLOC(M->ln1_w, (size_t)L * d); ALLOC(M->ln1_b, (size_t)L * d);
    ALLOC(M->qkvw, (size_t)L * 3 * d * d); ALLOC(M->qkvb, (size_t)L * 3 * d);
    ALLOC(M->outw, (size_t)L * d * d);     ALLOC(M->outb, (size_t)L * d);
    ALLOC(M->ln2_w, (size_t)L * d); ALLOC(M->ln2_b, (size_t)L * d);
    ALLOC(M->l1w, (size_t)L * ff * d); ALLOC(M->l1b, (size_t)L * ff);
    ALLOC(M->l2w, (size_t)L * d * ff); ALLOC(M->l2b, (size_t)L * d);
    ALLOC(M->fn_w, d); ALLOC(M->fn_b, d);
    ALLOC(M->h1w, (size_t)half * d); ALLOC(M->h1b, half);
    ALLOC(M->h2w, half);             ALLOC(M->h2b, 1);
    #undef ALLOC
    if (!ok) { seterr("内存分配失败"); fclose(f); free_model(M); return -1; }

    ok = ok && !rdf(f, M->proj_w, d) && !rdf(f, M->proj_b, d) && !rdf(f, M->pe, (size_t)W * d);
    for (int l = 0; ok && l < L; l++) {
        size_t lo = (size_t)l * d, lq = (size_t)l * 3 * d * d, lb = (size_t)l * 3 * d;
        size_t low = (size_t)l * d * d, lf1 = (size_t)l * ff * d, lf2 = (size_t)l * d * ff;
        ok = ok && !rdf(f, M->ln1_w + lo, d) && !rdf(f, M->ln1_b + lo, d) &&
             !rdf(f, M->qkvw + lq, (size_t)3 * d * d) && !rdf(f, M->qkvb + lb, (size_t)3 * d) &&
             !rdf(f, M->outw + low, (size_t)d * d) && !rdf(f, M->outb + (size_t)l * d, d) &&
             !rdf(f, M->ln2_w + lo, d) && !rdf(f, M->ln2_b + lo, d) &&
             !rdf(f, M->l1w + lf1, (size_t)ff * d) && !rdf(f, M->l1b + (size_t)l * ff, ff) &&
             !rdf(f, M->l2w + lf2, (size_t)d * ff) && !rdf(f, M->l2b + (size_t)l * d, d);
    }
    ok = ok && !rdf(f, M->fn_w, d) && !rdf(f, M->fn_b, d) &&
         !rdf(f, M->h1w, (size_t)half * d) && !rdf(f, M->h1b, half) &&
         !rdf(f, M->h2w, half) && !rdf(f, M->h2b, 1);
    fclose(f);
    if (!ok) { seterr("bundle 权重数据不完整"); free_model(M); return -1; }

    /* 工作区（双精度） */
    int ok2 = 1;
    #define ALLOCD(p, n) do { p = malloc(sizeof(double) * (size_t)(n)); ok2 = ok2 && (p || (n) == 0); } while (0)
    ALLOCD(M->xs, (size_t)W * d);
    ALLOCD(M->xbuf, (size_t)W * d);
    ALLOCD(M->qkv, (size_t)W * 3 * d);
    ALLOCD(M->av, (size_t)W * d);
    ALLOCD(M->ffn, (size_t)W * ff);
    ALLOCD(M->sc, (size_t)W * W);
    #undef ALLOCD
    if (!ok2) { seterr("工作区分配失败"); free_model(M); return -1; }
    return 0;
}

int nilm_edge_add_model(const char *bundle_path)
{
    if (!E.started) { seterr("须先调用 nilm_edge_start"); return -1; }
    if (E.n_models >= NILM_EDGE_MAX_MODELS) { seterr("模型数超出上限(8)"); return -1; }
    EdgeModel M;
    if (load_bundle(&M, bundle_path) != 0) return -1;
    if (!E.power_type_set) { E.power_type = M.power_type; E.power_type_set = 1; }
    else if (E.power_type != M.power_type) {
        seterr("power_type 与已加载模型不一致（apparent/active 不可混用）");
        free_model(&M); return -1;
    }
    E.models[E.n_models] = M;
    return E.n_models++;
}

/* ---------------- 推理核 ---------------- */

static double gelu1(double x)
{
    return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
}

static void ln_rows_d(double *dst, const double *src, const float *w, const float *b,
                      int rows, int d)
{
    for (int r = 0; r < rows; r++) {
        const double *s = src + (size_t)r * d;
        double *o = dst + (size_t)r * d;
        double m = 0.0, v = 0.0;
        for (int i = 0; i < d; i++) m += s[i];
        m /= d;
        for (int i = 0; i < d; i++) { double t = s[i] - m; v += t * t; }
        double inv = 1.0 / sqrt(v / d + LN_EPS);
        for (int i = 0; i < d; i++)
            o[i] = (s[i] - m) * inv * (double)w[i] + (double)b[i];
    }
}

/* dst[rows×out] = src[rows×in] @ W^T + b；W 布局 [out][in]（torch 同构） */
static void lin_rows_d(double *dst, const double *src, const float *w, const float *b,
                       int rows, int in, int out)
{
    for (int r = 0; r < rows; r++) {
        const double *s = src + (size_t)r * in;
        double *o = dst + (size_t)r * out;
        for (int j = 0; j < out; j++) {
            const float *wr = w + (size_t)j * in;
            double acc = b ? (double)b[j] : 0.0;
            for (int i = 0; i < in; i++) acc += s[i] * (double)wr[i];
            o[j] = acc;
        }
    }
}

/* 输入 win[window]（旧→新的 6s 功率），返回归一化预测值（中心点） */
static double model_forward(const EdgeModel *M, const double *win)
{
    const int W = M->window, d = M->d, L = M->num_layers, ff = M->ff;
    const int hd = M->head_dim;
    double *xs = M->xs, *xbuf = M->xbuf, *qkv = M->qkv, *av = M->av, *ffn = M->ffn, *sc = M->sc;

    for (int r = 0; r < W; r++) {
        double u = (win[r] - M->x_mean) / M->x_std; /* 归一化后转 float，与训练输入同精度 */
        float uf = (float)u;
        const float *pe = M->pe + (size_t)r * d;
        double *xr = xs + (size_t)r * d;
        for (int j = 0; j < d; j++)
            xr[j] = (double)uf * (double)M->proj_w[j] + (double)M->proj_b[j] + (double)pe[j];
    }

    for (int l = 0; l < L; l++) {
        size_t lo = (size_t)l * d;
        size_t lq = (size_t)l * 3 * d * d, lb = (size_t)l * 3 * d;
        size_t low = (size_t)l * d * d;
        size_t lf1 = (size_t)l * ff * d, lf2 = (size_t)l * d * ff;
        /* pre-LN 自注意力 */
        ln_rows_d(xbuf, xs, M->ln1_w + lo, M->ln1_b + lo, W, d);
        lin_rows_d(qkv, xbuf, M->qkvw + lq, M->qkvb + lb, W, d, 3 * d);
        double scale = 1.0 / sqrt((double)hd);
        for (int h = 0; h < M->nhead; h++) {
            int off = h * hd, koff = d + off, voff = 2 * d + off;
            for (int rq = 0; rq < W; rq++) {
                const double *q = qkv + (size_t)rq * 3 * d + off;
                double *scr = sc + (size_t)rq * W;
                double mx = -1e308;
                for (int rk = 0; rk < W; rk++) {
                    const double *k = qkv + (size_t)rk * 3 * d + koff;
                    double acc = 0.0;
                    for (int t = 0; t < hd; t++) acc += q[t] * k[t];
                    scr[rk] = acc * scale;
                    if (scr[rk] > mx) mx = scr[rk];
                }
                double sum = 0.0;
                for (int rk = 0; rk < W; rk++) {
                    scr[rk] = exp(scr[rk] - mx);
                    sum += scr[rk];
                }
                double inv = 1.0 / sum;
                double *arow = av + (size_t)rq * d + off;
                for (int t = 0; t < hd; t++) arow[t] = 0.0;
                for (int rk = 0; rk < W; rk++) {
                    double p = scr[rk] * inv;
                    const double *v = qkv + (size_t)rk * 3 * d + voff;
                    for (int t = 0; t < hd; t++) arow[t] += p * v[t];
                }
            }
        }
        lin_rows_d(xbuf, av, M->outw + low, M->outb + (size_t)l * d, W, d, d);
        for (size_t i = 0; i < (size_t)W * d; i++) xs[i] += xbuf[i];
        /* pre-LN 前馈 */
        ln_rows_d(xbuf, xs, M->ln2_w + lo, M->ln2_b + lo, W, d);
        lin_rows_d(ffn, xbuf, M->l1w + lf1, M->l1b + (size_t)l * ff, W, d, ff);
        for (size_t i = 0; i < (size_t)W * ff; i++) ffn[i] = gelu1(ffn[i]);
        lin_rows_d(xbuf, ffn, M->l2w + lf2, M->l2b + (size_t)l * d, W, ff, d);
        for (size_t i = 0; i < (size_t)W * d; i++) xs[i] += xbuf[i];
    }

    /* 中心行 → final norm → head */
    int c = W / 2;
    double crow[512];
    const double *srow = xs + (size_t)c * d;
    double m = 0.0, v = 0.0;
    for (int i = 0; i < d; i++) m += srow[i];
    m /= d;
    for (int i = 0; i < d; i++) { double t = srow[i] - m; v += t * t; }
    double inv = 1.0 / sqrt(v / d + LN_EPS);
    for (int i = 0; i < d; i++)
        crow[i] = (srow[i] - m) * inv * (double)M->fn_w[i] + (double)M->fn_b[i];

    int half = d / 2;
    double h1[512];
    for (int j = 0; j < half; j++) {
        const float *wr = M->h1w + (size_t)j * d;
        double acc = (double)M->h1b[j];
        for (int i = 0; i < d; i++) acc += crow[i] * (double)wr[i];
        h1[j] = gelu1(acc);
    }
    double y = (double)M->h2b[0];
    for (int i = 0; i < half; i++) y += h1[i] * (double)M->h2w[i];
    return y;
}

/* ---------------- 6s 桶引擎 ---------------- */

static void infer_one(EdgeModel *M)
{
    if (E.raw_count < M->window) return;
    const int W = M->window;
    double win[RAW_RING_MAX];
    int64_t newest = E.raw_count - 1;
    for (int k = 0; k < W; k++) {
        int64_t abs = newest - W + 1 + k;
        win[k] = E.raw_val[abs % RAW_RING_MAX];
    }
    double pred = model_forward(M, win) * M->y_std + M->y_mean;
    int64_t cabs = newest - W + 1 + W / 2; /* 中心样本绝对索引 */
    NilmEdgeResult *r = &M->rq[M->r_tail % NILM_EDGE_RESULT_RING];
    r->center_ts = E.raw_ts[cabs % RAW_RING_MAX];
    r->aggregate_w = E.raw_val[cabs % RAW_RING_MAX];
    r->pred_w = pred;
    r->on = pred >= M->threshold ? 1 : 0;
    r->reserved = 0;
    M->r_tail++;
    if (M->r_tail - M->r_head > NILM_EDGE_RESULT_RING)
        M->r_head = M->r_tail - NILM_EDGE_RESULT_RING; /* 溢出丢最旧 */
}

static void push_value(double v, double ts)
{
    E.raw_val[E.raw_count % RAW_RING_MAX] = v;
    E.raw_ts[E.raw_count % RAW_RING_MAX] = ts;
    E.raw_count++;
    E.last_value = v;
    E.have_last = 1;
    E.carry_run = 0;
    for (int i = 0; i < E.n_models; i++)
        infer_one(&E.models[i]);
}

/* 结束当前桶（cur_bucket）：有数据→按周波均值换算三相功率；空桶→用上一桶值填充
 * （等价训练期 ffill）；连续空桶超 30min→判长缺口，清缓冲重置暖机。 */
static void finalize_bucket(void)
{
    int have = 0;
    double v = 0.0;
    if (E.bucket_n > 0) {
        double n = (double)E.bucket_n;
        for (int ph = 0; ph < 3; ph++) {
            double P = E.sum_ui[ph] / n;
            double Vrms = sqrt(E.sum_u2[ph] / n);
            double Irms = sqrt(E.sum_i2[ph] / n);
            double S = Vrms * Irms;
            v += (E.power_type == 0) ? S : P;
        }
        have = 1;
    } else if (E.have_last) {
        E.carry_run++;
        if (E.carry_run > CARRY_LIMIT) {
            /* 长缺口：等价制备期 long-gap 剔除，缓冲清零重置暖机 */
            E.raw_count = 0;
            E.have_last = 0;
            E.carry_run = 0;
            have = 0;
        } else {
            v = E.last_value;
            have = 1;
        }
    }
    if (have)
        push_value(v, (double)E.cur_bucket * TICK_SEC);
    E.sum_ui[0] = E.sum_ui[1] = E.sum_ui[2] = 0.0;
    E.sum_u2[0] = E.sum_u2[1] = E.sum_u2[2] = 0.0;
    E.sum_i2[0] = E.sum_i2[1] = E.sum_i2[2] = 0.0;
    E.bucket_n = 0;
}

/* ---------------- 公开 API ---------------- */

int nilm_edge_push_packet(const float *wave, int points, int channels, double ts)
{
    if (!E.started) { seterr("须先调用 nilm_edge_start"); return -1; }
    if (!wave || channels != 6 || points <= 0) {
        seterr("参数非法：wave 非空 / channels=6 / points>0");
        return -1;
    }
    int64_t b = (int64_t)floor(ts / TICK_SEC);
    if (!E.have_bucket) {
        E.cur_bucket = b;
        E.have_bucket = 1;
    } else if (b > E.cur_bucket) {
        finalize_bucket(); /* 结束旧桶（含已累计数据） */
        for (int64_t k = E.cur_bucket + 1; k < b; k++) {
            E.cur_bucket = k;
            finalize_bucket(); /* 跳过的空桶：填充/长缺口 */
        }
        E.cur_bucket = b;
    }
    /* b <= cur_bucket（迟到包/同桶包）：并入当前桶累计（调用方保证 ts 单调不减） */
    for (int p = 0; p < points; p++) {
        const float *s = wave + (size_t)p * channels;
        for (int ph = 0; ph < 3; ph++) {
            double u = (double)s[ph * 2];
            double i = (double)s[ph * 2 + 1];
            E.sum_ui[ph] += u * i;
            E.sum_u2[ph] += u * u;
            E.sum_i2[ph] += i * i;
        }
    }
    E.bucket_n += points;
    return 0;
}

int nilm_edge_poll(int model_id, NilmEdgeResult *out)
{
    if (!E.started || model_id < 0 || model_id >= E.n_models) {
        seterr("model_id 无效");
        return -1;
    }
    EdgeModel *M = &E.models[model_id];
    if (M->r_head >= M->r_tail)
        return 0;
    if (out)
        *out = M->rq[M->r_head % NILM_EDGE_RESULT_RING];
    M->r_head++;
    return 1;
}

int nilm_edge_flush(void)
{
    if (!E.started) { seterr("须先调用 nilm_edge_start"); return -1; }
    if (E.have_bucket && E.bucket_n > 0)
        finalize_bucket();
    E.have_bucket = 0;
    return 0;
}
