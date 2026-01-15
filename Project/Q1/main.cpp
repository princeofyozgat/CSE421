#define UART_MODE 1

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdarg>
#include <cstring>

#include "mbed.h"
#include "cmsis.h"

// ====== DISCO_F746NG SDRAM BSP ======
extern "C" {
#include "BSP/32F746G-Discovery/stm32746g_discovery_sdram.h"
}

// ====== TFLM ======
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/memory_helpers.h"

// Test input
#include "src/model_data.h"

// -----------------------------
// SDRAM base (FMC mapped)
// -----------------------------
static constexpr uintptr_t kSdramBase = 0xC0000000u;
// base'in hemen başına yazmayalım, 4KB offset verelim:
static constexpr uintptr_t kSdramArenaBase = kSdramBase + 0x1000u;

// Arena boyutu (SDRAM'e koyduğumuz için büyük tutabiliriz)
constexpr int kTensorArenaSize = 900 * 1024; // 900KB

// -----------------------------
// UART (PC <-> MCU)
// -----------------------------
static BufferedSerial pc(USBTX, USBRX, 115200);

#if UART_MODE
// "tam yaz" helper (kısmi write olabiliyor => tamamını gönder)
static void uart_write_all(const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = pc.write(data + sent, len - sent);
        if (w > 0) sent += (size_t)w;
        // blokluyken genelde gerek kalmaz, ama güvenli kalsın
    }
}

// printf benzeri: TEK buffer -> TEK write
static void uart_printf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if (n > (int)sizeof(buf)) n = (int)sizeof(buf);
    uart_write_all(buf, (size_t)n);
}

// Tag’li satır basma
static void log_line(const char* fmt, ...) {
    char msg[448];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    uart_printf("@LOG %s\n", msg);
}

static void ready_line_json(const char* json_no_newline) {
    uart_printf("@READY %s\n", json_no_newline);
}

static void res_line_json(const char* json_no_newline) {
    uart_printf("@RES %s\n", json_no_newline);
}

static void err_line_json(const char* json_no_newline) {
    uart_printf("@ERR %s\n", json_no_newline);
}
#endif // UART_MODE


// Güvenli okuma helper'ı
static bool read_exact(uint8_t* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        if (pc.readable()) {
            ssize_t r = pc.read(dst + got, n - got);
            if (r > 0) got += (size_t)r;
        }
    }
    return true;
}

// write_str: UART_MODE=1 iken "tam yaz" kullan
static void write_str(const char* s) {
#if UART_MODE
    uart_write_all(s, strlen(s));
#else
    pc.write(s, strlen(s));
#endif
}

static void disable_unaligned_trap_and_print() {
    // UNALIGN_TRP kapat
    SCB->CCR &= ~SCB_CCR_UNALIGN_TRP_Msk;
    __DSB();
    __ISB();

#if !UART_MODE
    printf("CCR=0x%08lX UNALIGN_TRP=%lu\r\n",
           (unsigned long)SCB->CCR,
           (unsigned long)((SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk) ? 1 : 0));
#else
    // UART_MODE'da istersen log basabilirsin:
    log_line("CCR=0x%08lX UNALIGN_TRP=%lu",
             (unsigned long)SCB->CCR,
             (unsigned long)((SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk) ? 1 : 0));
#endif
}


// IoU / NMS
struct Det { float ymin, xmin, ymax, xmax; int cls; float score; };

static float iou(const Det& a, const Det& b) {
  float iymin = std::max(a.ymin, b.ymin);
  float ixmin = std::max(a.xmin, b.xmin);
  float iymax = std::min(a.ymax, b.ymax);
  float ixmax = std::min(a.xmax, b.xmax);
  float ih = std::max(0.0f, iymax - iymin);
  float iw = std::max(0.0f, ixmax - ixmin);
  float inter = ih * iw;
  float area_a = std::max(0.0f, a.ymax - a.ymin) * std::max(0.0f, a.xmax - a.xmin);
  float area_b = std::max(0.0f, b.ymax - b.ymin) * std::max(0.0f, b.xmax - b.xmin);
  float uni = area_a + area_b - inter;
  if (uni <= 1e-6f) return 0.0f;
  return inter / uni;
}

static int nms(Det* dets, int count, float iou_thresh, int max_keep) {
  std::sort(dets, dets + count, [](const Det& a, const Det& b){ return a.score > b.score; });
  bool sup[512] = {false};
  int keep = 0;
  for (int i = 0; i < count && keep < max_keep; i++) {
    if (sup[i]) continue;
    dets[keep++] = dets[i];
    for (int j = i + 1; j < count; j++) {
      if (sup[j]) continue;
      if (dets[i].cls != dets[j].cls) continue;
      if (iou(dets[i], dets[j]) > iou_thresh) sup[j] = true;
    }
  }
  return keep;
}

// -----------------------------
// Fixed-point helpers
// -----------------------------
static void split_fix(float v, int decimals, long* ip, long* fp) {
    bool neg = (v < 0);
    if (neg) v = -v;

    int64_t mul = 1;
    for (int i = 0; i < decimals; i++) mul *= 10;

    int64_t scaled = (int64_t)llroundf(v * (float)mul);
    int64_t i_part = scaled / mul;
    int64_t f_part = scaled % mul;

    if (neg) i_part = -i_part;

    *ip = (long)i_part;
    *fp = (long)f_part;
}

static long scale_x1e6(float s) {
    return (long)llroundf(s * 1000000.0f);
}

static inline int8_t quant_u8_to_i8(uint8_t u, float scale, int zp) {
  int q = (int)lrintf(((float)u) / scale) + zp;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

static bool sdram_quick_test() {
  volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(kSdramBase);

  p[0] = 0xA5A5A5A5u;
  p[1] = 0x5A5A5A5Au;

  uint32_t r0 = p[0];
  uint32_t r1 = p[1];

#if !UART_MODE
  printf("SDRAM test readback: %08lx %08lx\n", (unsigned long)r0, (unsigned long)r1);
#else
  log_line("SDRAM test readback: %08lx %08lx", (unsigned long)r0, (unsigned long)r1);
#endif

  return (r0 == 0xA5A5A5A5u) && (r1 == 0x5A5A5A5Au);
}

static bool sdram_block_test(uint32_t words) {
  volatile uint32_t* p = (volatile uint32_t*)kSdramArenaBase;
  for (uint32_t i = 0; i < words; i++) p[i] = 0xA5A50000u ^ i;
  __DSB(); __ISB();

  for (uint32_t i = 0; i < words; i++) {
    uint32_t exp = 0xA5A50000u ^ i;
    if (p[i] != exp) {
#if !UART_MODE
      printf("SDRAM mismatch @%lu got=%08lx exp=%08lx\n",
             (unsigned long)i, (unsigned long)p[i], (unsigned long)exp);
#else
      log_line("SDRAM mismatch @%lu got=%08lx exp=%08lx",
               (unsigned long)i, (unsigned long)p[i], (unsigned long)exp);
#endif
      return false;
    }
  }
  return true;
}

#pragma pack(push, 1)
struct ImgHeader {
    uint8_t magic0;   // 'I'
    uint8_t magic1;   // 'M'
    uint16_t h;       // little-endian
    uint16_t w;       // little-endian
    uint8_t c;        // 1
    uint8_t dtype;    // 0 = uint8
    uint32_t len;     // h*w*c
};
#pragma pack(pop)

static bool recv_image(std::unique_ptr<uint8_t[]>& buf, uint16_t& H, uint16_t& W, uint8_t& C) {
    // 1) magic sync: 'I''M'
    uint8_t b = 0;
    while (true) {
        read_exact(&b, 1);
        if (b != 'I') continue;
        read_exact(&b, 1);
        if (b == 'M') break;
    }

    // 2) header kalanını oku
    ImgHeader hdr;
    hdr.magic0 = 'I';
    hdr.magic1 = 'M';
    read_exact(((uint8_t*)&hdr) + 2, sizeof(ImgHeader) - 2);

    H = hdr.h;
    W = hdr.w;
    C = hdr.c;

    if (hdr.dtype != 0) {
#if UART_MODE
        err_line_json("{\"err\":\"dtype_not_u8\"}");
#else
        write_str("{\"err\":\"dtype_not_u8\"}\n");
#endif
        return false;
    }
    if (hdr.len != (uint32_t)hdr.h * (uint32_t)hdr.w * (uint32_t)hdr.c) {
#if UART_MODE
        err_line_json("{\"err\":\"len_mismatch\"}");
#else
        write_str("{\"err\":\"len_mismatch\"}\n");
#endif
        return false;
    }
    if (hdr.len > (1024u * 1024u)) {
#if UART_MODE
        err_line_json("{\"err\":\"too_big\"}");
#else
        write_str("{\"err\":\"too_big\"}\n");
#endif
        return false;
    }

    buf.reset(new uint8_t[hdr.len]);
    read_exact(buf.get(), hdr.len);
    return true;
}

int main() {
#if UART_MODE
  pc.set_format(8, BufferedSerial::None, 1);
  pc.set_blocking(true);
  ThisThread::sleep_for(50ms);
  log_line("=== TinySSD TFLM (DISCO_F746NG) ===");
#else
  printf("=== TinySSD TFLM (DISCO_F746NG) ===\n");
#endif

  // Cache disable
#if !UART_MODE
  printf("Caches disabled\n");
#else
  log_line("Caches disabled");
#endif

  SCB_DisableDCache();
  SCB_DisableICache();
  __DSB(); __ISB();

  // 1) SDRAM init
#if UART_MODE
  log_line("Init SDRAM...");
#else
  printf("Init SDRAM...\n");
#endif

  if (BSP_SDRAM_Init() != 0) {
#if UART_MODE
    err_line_json("{\"err\":\"sdram_init\"}");
#else
    printf("BSP_SDRAM_Init FAILED!\n");
#endif
    while (true) { ThisThread::sleep_for(1000ms); }
  }

#if UART_MODE
  log_line("SDRAM init OK");
#else
  printf("SDRAM init OK\n");
#endif

  // 2) SDRAM test
  if (!sdram_quick_test()) {
#if UART_MODE
    err_line_json("{\"err\":\"sdram_quick_test\"}");
#else
    printf("SDRAM test FAILED (write/read mismatch)!\n");
#endif
    while (true) { ThisThread::sleep_for(1000ms); }
  }

#if UART_MODE
  log_line("SDRAM quick test OK");
  log_line("SDRAM block test...");
#else
  printf("SDRAM test OK\n");
  printf("SDRAM block test...\n");
#endif

  if (!sdram_block_test(32 * 1024)) { // 32K word = 128KB
#if UART_MODE
    err_line_json("{\"err\":\"sdram_block_test\"}");
#else
    printf("SDRAM block test FAILED!\n");
#endif
    while (true) { ThisThread::sleep_for(1000ms); }
  }

#if UART_MODE
  log_line("SDRAM block test OK");
#else
  printf("SDRAM block test OK\n");
#endif

  // 3) TFLM setup
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

#if UART_MODE
  log_line("Model ptr=%p (mod16=%lu)",
           mobilenet_ssd_int8_tflite,
           (unsigned long)((uintptr_t)mobilenet_ssd_int8_tflite & 0xF));
#else
  printf("Model ptr = %p (mod16=%lu)\n",
         mobilenet_ssd_int8_tflite,
         (unsigned long)((uintptr_t)mobilenet_ssd_int8_tflite & 0xF));
#endif

  const tflite::Model* model = tflite::GetModel(mobilenet_ssd_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
#if UART_MODE
    err_line_json("{\"err\":\"schema_mismatch\"}");
#else
    printf("Schema mismatch!\n");
#endif
    while (true) { ThisThread::sleep_for(1000ms); }
  }

  static tflite::AllOpsResolver resolver;

  // Arena: SDRAM
  uint8_t* tensor_arena = reinterpret_cast<uint8_t*>(kSdramArenaBase);

#if UART_MODE
  log_line("Tensor arena @ 0x%08lx size=%d", (unsigned long)kSdramArenaBase, kTensorArenaSize);
#else
  printf("Tensor arena @ 0x%08lx size=%d\n", (unsigned long)kSdramArenaBase, kTensorArenaSize);
#endif

  static tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

#if UART_MODE
  log_line("Before AllocateTensors CCR=0x%08lX UNALIGN_TRP=%lu",
           (unsigned long)SCB->CCR,
           (unsigned long)((SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk) ? 1 : 0));
  log_line("Before AllocateTensors");
#else
  printf("Before AllocateTensors CCR=0x%08lX UNALIGN_TRP=%lu\n",
         (unsigned long)SCB->CCR,
         (unsigned long)((SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk) ? 1 : 0));
  printf("Before AllocateTensors\n");
#endif

  auto st_alloc = interpreter.AllocateTensors();

#if UART_MODE
  log_line("After AllocateTensors st=%d", (int)st_alloc);
  log_line("outputs_size=%d", interpreter.outputs_size());
#else
  printf("After AllocateTensors st=%d\n", (int)st_alloc);
  printf("outputs_size=%d\n", interpreter.outputs_size());
#endif

  for (int oi = 0; oi < interpreter.outputs_size(); oi++) {
    TfLiteTensor* t = interpreter.output(oi);
#if UART_MODE
    // dims'i kısaca yazalım (çok uzatmak istemiyorsan kapatabilirsin)
    log_line("out[%d] type=%d bytes=%d zp=%ld scale_x1e6=%ld",
             oi, (int)t->type, (int)t->bytes,
             (long)t->params.zero_point,
             (long)scale_x1e6(t->params.scale));
#else
    printf("out[%d] type=%d dims=", oi, t->type);
    for (int d = 0; d < t->dims->size; d++) printf("%d ", t->dims->data[d]);
    printf(" bytes=%d zp=%ld scale_x1e6=%ld\n",
           t->bytes,
           (long)t->params.zero_point,
           (long)scale_x1e6(t->params.scale));
#endif
  }

  TfLiteTensor* input = interpreter.input(0);

  // Input dims
  int inH = input->dims->data[1];
  int inW = input->dims->data[2];
  int inC = input->dims->data[3];

#if UART_MODE
  // READY mesajı: PC bunu bekleyip sonra image göndermeli
  char ready_json[128];
  snprintf(ready_json, sizeof(ready_json),
           "{\"ready\":true,\"inH\":%d,\"inW\":%d,\"inC\":%d}", inH, inW, inC);
  ready_line_json(ready_json);
  #if UART_MODE
  // READY'yi PC kaçırmasın diye, ilk byte gelene kadar periyodik tekrar gönder
  Timer t;
  t.start();
  while (!pc.readable()) {
      if (t.elapsed_time() >= 1000ms) {
          ready_line_json(ready_json);   // tekrar gönder
          t.reset();
      }
      ThisThread::sleep_for(10ms);
  }
    #endif
#else
  char msg[128];
  snprintf(msg, sizeof(msg),
           "{\"ready\":true,\"inH\":%d,\"inW\":%d,\"inC\":%d}\n", inH, inW, inC);
  write_str(msg);
#endif

  while (true) {
    std::unique_ptr<uint8_t[]> img;
    uint16_t H=0, W=0; uint8_t C=0;

    if (!recv_image(img, H, W, C)) {
      continue;
    }

    if ((int)H != inH || (int)W != inW || (int)C != inC) {
      char e[128];
      snprintf(e, sizeof(e),
               "{\"err\":\"shape\",\"got\":[%u,%u,%u],\"need\":[%d,%d,%d]}",
               H, W, C, inH, inW, inC);
#if UART_MODE
      err_line_json(e);
#else
      // eski davranış
      strcat(e, "\n");
      write_str(e);
#endif
      continue;
    }

    if (input->type != kTfLiteInt8) {
#if UART_MODE
      err_line_json("{\"err\":\"input_not_int8\"}");
#else
      write_str("{\"err\":\"input_not_int8\"}\n");
#endif
      continue;
    }

    // input quantize
    int8_t* in = input->data.int8;
    float s = input->params.scale;
    int zp  = input->params.zero_point;

    const uint32_t N = (uint32_t)inH * (uint32_t)inW * (uint32_t)inC;
    for (uint32_t i = 0; i < N; i++) {
      in[i] = quant_u8_to_i8(img[i], s, zp);
    }

    // invoke
    TfLiteStatus st = interpreter.Invoke();
    if (st != kTfLiteOk) {
#if UART_MODE
      err_line_json("{\"err\":\"invoke\"}");
#else
      write_str("{\"err\":\"invoke\"}\n");
#endif
      continue;
    }

    // outputs
    TfLiteTensor* out_boxes  = interpreter.output(1);
    TfLiteTensor* out_scores = interpreter.output(0);

    int Ndet = out_scores->dims->data[1];
    int Kcls = out_scores->dims->data[2];

    int best_i = 0;
    int best_c = 0;
    float best_sc = -1e9f;

    auto dequant_i8_local = [](int8_t v, float scale, int zp) {
      return ((int)v - zp) * scale;
    };

    if (out_scores->type == kTfLiteInt8) {
      const int8_t* s8 = out_scores->data.int8;
      float sS = out_scores->params.scale;
      int   sZ = out_scores->params.zero_point;

      for (int i = 0; i < Ndet; i++) {
        for (int c = 0; c < Kcls; c++) {
          float sc = dequant_i8_local(s8[i*Kcls + c], sS, sZ);
          if (sc > best_sc) { best_sc = sc; best_i = i; best_c = c; }
        }
      }
    } else if (out_scores->type == kTfLiteFloat32) {
      const float* sf = out_scores->data.f;
      for (int i = 0; i < Ndet; i++) {
        for (int c = 0; c < Kcls; c++) {
          float sc = sf[i*Kcls + c];
          if (sc > best_sc) { best_sc = sc; best_i = i; best_c = c; }
        }
      }
    } else {
#if UART_MODE
      err_line_json("{\"err\":\"scores_type\"}");
#else
      write_str("{\"err\":\"scores_type\"}\n");
#endif
      continue;
    }

    float ymin=0,xmin=0,ymax=0,xmax=0;
    if (out_boxes->type == kTfLiteInt8) {
      const int8_t* b8 = out_boxes->data.int8;
      float bS = out_boxes->params.scale;
      int   bZ = out_boxes->params.zero_point;
      ymin = dequant_i8_local(b8[best_i*4 + 0], bS, bZ);
      xmin = dequant_i8_local(b8[best_i*4 + 1], bS, bZ);
      ymax = dequant_i8_local(b8[best_i*4 + 2], bS, bZ);
      xmax = dequant_i8_local(b8[best_i*4 + 3], bS, bZ);
    } else if (out_boxes->type == kTfLiteFloat32) {
      const float* bf = out_boxes->data.f;
      ymin = bf[best_i*4 + 0];
      xmin = bf[best_i*4 + 1];
      ymax = bf[best_i*4 + 2];
      xmax = bf[best_i*4 + 3];
    } else {
#if UART_MODE
      err_line_json("{\"err\":\"boxes_type\"}");
#else
      write_str("{\"err\":\"boxes_type\"}\n");
#endif
      continue;
    }

    long score_i=0, score_f=0;
    long ymin_i=0, ymin_f=0, xmin_i=0, xmin_f=0, ymax_i=0, ymax_f=0, xmax_i=0, xmax_f=0;

    split_fix(best_sc, 6, &score_i, &score_f);
    split_fix(ymin,   6, &ymin_i, &ymin_f);
    split_fix(xmin,   6, &xmin_i, &xmin_f);
    split_fix(ymax,   6, &ymax_i, &ymax_f);
    split_fix(xmax,   6, &xmax_i, &xmax_f);

    char out[320];
    snprintf(out, sizeof(out),
             "{\"ok\":true,\"best_i\":%d,\"cls\":%d,"
             "\"score_i\":%ld,\"score_f\":%ld,"
             "\"box_i\":[%ld,%ld,%ld,%ld],"
             "\"box_f\":[%ld,%ld,%ld,%ld]}",
             best_i, best_c,
             score_i, score_f,
             ymin_i, xmin_i, ymax_i, xmax_i,
             ymin_f, xmin_f, ymax_f, xmax_f);

#if UART_MODE
    res_line_json(out);
#else
    // eski davranış: newline ile raw JSON
    strcat(out, "\n");
    write_str(out);
#endif
  }
}