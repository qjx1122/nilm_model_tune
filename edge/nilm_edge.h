/* nilm_edge.h — NILM 边缘实时推理引擎（P0，C ABI 动态库）
 *
 * 形态：终端主程序加载本库，按周波包推送采集数据，轮询 6s 粒度推理结果。
 * 数据流：128 点×6 通道周波包（20ms/包）→ 每周波电参量 → 6s 桶聚合 →
 *        环形缓冲（window 点）→ Transformer seq2point 前向 → pred(W)/ON。
 *
 * 接口约定（docs/EDGE_DEPLOYMENT.md §1，2026-09-16 与终端方锁定）：
 *   - 传输=动态库调用；数据=工程量 float（V/A），带时间戳；
 *   - 每包 128 点×6 通道（一包=一个整周波，周波对齐/频率跟踪由终端采集侧负责）；
 *   - 丢帧由调用方用上一包数据重推（本库按时间戳分桶，兼容该语义）；
 *   - 6 通道同采同时钟，交错顺序 [uA, iA, uB, iB, uC, iC]。
 *
 * 线程模型：P0 非线程安全——所有调用须来自同一线程。
 * 延迟：中心窗语义（与训练一致），预测中心落后最新数据 (window - window/2 - 1)×6s
 *      （window=192 → 570s；window=96 → 282s）。暖机：开机后 window 个 6s 桶内无结果。
 */
#ifndef NILM_EDGE_H
#define NILM_EDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 符号导出：构建 DLL 时定义 NILM_EDGE_BUILDING → dllexport；消费方包含本头 →
 * dllimport（Windows 链接导入库）；Linux/macOS 下为空。 */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef NILM_EDGE_BUILDING
    #define NILM_EDGE_API __declspec(dllexport)
  #else
    #define NILM_EDGE_API __declspec(dllimport)
  #endif
#else
  #define NILM_EDGE_API
#endif

#define NILM_EDGE_MAX_MODELS 8
#define NILM_EDGE_RESULT_RING 2048 /* 每模型结果 FIFO 深度（6s 结果条数）。\n * 2048≈3.4h 不轮询容忍；长缺口(>30min)恢复包会瞬时 carry 入队 ~300 条，256 会溢出丢最旧 */

typedef struct {
    double center_ts;   /* 预测中心点时间戳（秒，= 该 6s 桶起点） */
    double aggregate_w; /* 中心点所在 6s 桶的三相总功率（W，bundle 口径） */
    double pred_w;      /* 模型预测的电器功率（W，可为负=回归输出原值） */
    int    on;          /* pred_w >= on_threshold_watts */
    int    reserved;    /* 对齐保留 */
} NilmEdgeResult;

/* 初始化引擎（进程一次）。返回 0=成功，<0=失败。 */
NILM_EDGE_API int  nilm_edge_start(void);

/* 加载部署包（export_edge_bundle.py 导出的 model.bin），返回 model_id（0..N-1），<0=失败。
 * 多模型（kettle/dw/mw…）共享同一功率流，各自独立 window/统计量/阈值；power_type 须一致。 */
NILM_EDGE_API int  nilm_edge_add_model(const char *bundle_path);

/* 推入一包周波数据。
 * wave：points×channels 交错浮点工程量，通道顺序 [uA,iA,uB,iB,uC,iC]；
 * points：本包采样点数（典型 128）；channels：必须为 6；
 * ts：本包时间戳（秒，单调不减；丢帧由调用方按协议用上一包数据填充后重推）。
 * 返回 0=成功，<0=失败（见 nilm_edge_last_error）。 */
NILM_EDGE_API int  nilm_edge_push_packet(const float *wave, int points, int channels, double ts);

/* 轮询模型结果（FIFO）：按时间顺序取出最早一条未读结果填 *out 并返回 1；
 * 无未读返回 0；<0=错误。循环调用直到返回 0 可取空（不丢 6s 事件段）；
 * 积压超过 2048 条时丢弃最旧。 */
NILM_EDGE_API int  nilm_edge_poll(int model_id, NilmEdgeResult *out);

/* 提前结束当前未满 6s 的桶（可选：停机/长间隔前调用，触发一次最终聚合+推理）。 */
NILM_EDGE_API int  nilm_edge_flush(void);

/* 释放全部资源。 */
NILM_EDGE_API void nilm_edge_shutdown(void);

NILM_EDGE_API const char *nilm_edge_version(void);
NILM_EDGE_API const char *nilm_edge_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NILM_EDGE_H */
