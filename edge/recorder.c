/* recorder.c — 周波数据录制器实现（C99，零外部依赖，stdio 缓冲写）。 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "recorder.h"

#define RAW_MAGIC   "NILMRAW1"
#define RAW_VERSION 1
#define RAW_HDR_SZ  64

static struct {
    FILE *f;
    int points, channels;
    char err[256];
} R;

const char *nilm_rec_last_error(void) { return R.err; }
int nilm_rec_is_open(void) { return R.f != NULL; }

static void errset(const char *msg)
{
    snprintf(R.err, sizeof(R.err), "%s", msg);
}

int nilm_rec_open(const char *path, int points, int channels)
{
    if (R.f) { errset("已有录制打开：先 nilm_rec_close 再 open"); return -1; }
    if (!path || points <= 0 || channels <= 0 || points > 4096 || channels > 64) {
        errset("参数非法：path 非空 / 0<points<=4096 / 0<channels<=64");
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) { errset("录制文件创建失败（路径/权限？）"); return -1; }
    char hdr[RAW_HDR_SZ];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, RAW_MAGIC, 8);
    int32_t ints[4] = { RAW_VERSION, (int32_t)points, (int32_t)channels, 0 };
    memcpy(hdr + 8, ints, sizeof(ints));
    double dbls[3] = { 6400.0, 50.0, 0.0 };   /* 信息性：采样率/工频（转换器写入 meta） */
    memcpy(hdr + 24, dbls, sizeof(dbls));
    /* hdr[48..64) 保留 0 */
    if (fwrite(hdr, 1, RAW_HDR_SZ, f) != RAW_HDR_SZ) {
        errset("头写入失败（磁盘满？）");
        fclose(f);
        return -1;
    }
    R.f = f;
    R.points = points;
    R.channels = channels;
    return 0;
}

int nilm_rec_packet(const float *wave, int points, int channels, double ts)
{
    if (!R.f) { errset("录制未打开：先 nilm_rec_open"); return -1; }
    if (!wave || points != R.points || channels != R.channels) {
        errset("包参数与 open 时不一致（points/channels 须固定）");
        return -1;
    }
    size_t n = (size_t)points * (size_t)channels;
    if (fwrite(wave, sizeof(float), n, R.f) != n) {
        errset("波形写入失败（磁盘满？）");
        return -1;
    }
    if (fwrite(&ts, sizeof(double), 1, R.f) != 1) {
        errset("时间戳写入失败（磁盘满？）");
        return -1;
    }
    return 0;
}

int nilm_rec_close(void)
{
    if (!R.f)
        return 0;
    int rc = 0;
    if (fflush(R.f) != 0) {
        errset("冲刷失败");
        rc = -1;
    }
    if (fclose(R.f) != 0) {
        errset("关闭失败（数据可能未完整落盘）");
        rc = -1;
    }
    R.f = NULL;
    return rc;
}
