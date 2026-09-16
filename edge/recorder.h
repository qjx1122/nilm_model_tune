/* recorder.h — 周波数据录制器（现场录制落盘，配合 nilm_edge 引擎双写）。
 *
 * 用途：终端主程序在向 nilm_edge_push_packet 推送每包数据的同时，调用本录制器把
 * 同一包数据 append 到 .raw 文件——一条代码路径同时完成「实时推理」与「现场录制」，
 * 录制文件经 scripts/raw_to_recnpz.py 转为 NPZ v1 后即可用 edge_stream_test.py --mode
 * replay 回放（S2 现场验证 / 黄金集留存）。
 *
 * .raw 格式 v1（小端）：
 *   头 64B：magic "NILMRAW1"(8) | int32×4 [version=1, points, channels, 保留]
 *           | double×3 [fs=6400, f0=50, 保留] | 保留 16B（全 0）
 *   记录（定长，append）：wave float32 × points*channels + ts float64 × 1
 *   （points=128, channels=6 时记录长 3080B；截断尾记录可按长度识别并丢弃）
 *
 * 线程模型：与引擎一致——全部调用须来自同一线程（建议就在推送点调用）。
 */
#ifndef NILM_RECORDER_H
#define NILM_RECORDER_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef NILM_REC_BUILDING
    #define NILM_REC_API __declspec(dllexport)
  #else
    #define NILM_REC_API __declspec(dllimport)
  #endif
#else
  #define NILM_REC_API
#endif

/* 打开录制文件并写头（points/channels 为后续每包的固定尺寸）。
 * 返回 0=成功，<0=失败（已有录制打开 / 参数非法 / 文件打不开）。 */
int NILM_REC_API nilm_rec_open(const char *path, int points, int channels);

/* 追加一包（与 nilm_edge_push_packet 同参语义；points/channels 须与 open 时一致）。
 * 返回 0=成功，<0=失败（见 nilm_rec_last_error）。 */
int NILM_REC_API nilm_rec_packet(const float *wave, int points, int channels, double ts);

/* 冲刷并关闭（停机前必调；漏调由 fclose 兜底但有丢尾包风险）。返回 0=成功。 */
int NILM_REC_API nilm_rec_close(void);

int NILM_REC_API nilm_rec_is_open(void);
const char *NILM_REC_API nilm_rec_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NILM_RECORDER_H */
