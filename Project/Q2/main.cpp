// main.cpp (DISCO_F746NG + Mbed OS + TFLite Micro) - REQ/RESP UART protocol
#include "mbed.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "src/model_data.h"

// ====== DISCO_F746NG SDRAM BSP ======
extern "C" {
#include "BSP/32F746G-Discovery/stm32746g_discovery_sdram.h"
}

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===================== Config =====================
static constexpr int   kImgW = 28;
static constexpr int   kImgH = 28;
static constexpr int   kImgSize = kImgW * kImgH;     // 784
static constexpr int   kNumClasses = 10;

static constexpr size_t kTensorArenaSize = 310 * 1024;

// UART
static constexpr int kBaud = 115200;

// SDRAM base (STM32F746 DISCO: external SDRAM genelde 0xC0000000)
static constexpr uint32_t kSdramBase = 0xC0000000u;
static constexpr uint32_t kSdramSizeBytes = 8u * 1024u * 1024u; // çoğu kartta 8MB

// ===================== Simple SDRAM allocator =====================
static uint32_t g_sdram_offset = 0;

static void* sdram_alloc(size_t bytes, size_t align) {
    uint32_t off = g_sdram_offset;
    uint32_t mask = (uint32_t)(align - 1);
    off = (off + mask) & ~mask;

    if (off + bytes > kSdramSizeBytes) return nullptr;

    void* p = (void*)(kSdramBase + off);
    g_sdram_offset = off + (uint32_t)bytes;
    return p;
}

// ===================== UART helpers =====================
static UnbufferedSerial g_serial(USBTX, USBRX, kBaud);

static bool read_exact(uint8_t* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = g_serial.read(dst + got, n - got);
        if (r > 0) {
            got += (size_t)r;
        } else {
            ThisThread::sleep_for(1ms); // non-blocking read -> bekle
        }
    }
    return true;
}

static void write_all(const uint8_t* src, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = g_serial.write(src + sent, n - sent);
        if (w > 0) {
            sent += (size_t)w;
        } else {
            ThisThread::sleep_for(1ms);
        }
    }
}

static uint32_t read_u32_le(const uint8_t b[4]) {
    return (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

static void write_u32_le(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v & 0xFF);
    out[1] = (uint8_t)((v >> 8) & 0xFF);
    out[2] = (uint8_t)((v >> 16) & 0xFF);
    out[3] = (uint8_t)((v >> 24) & 0xFF);
}



// ===================== Quant / Argmax =====================
static int8_t quantize_01_to_int8(float x01, float scale, int zero_point) {
    int32_t q = (int32_t)lrintf(x01 / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    return (int8_t)q;
}

static int argmax_int8(const int8_t* data, int len) {
    int best_i = 0;
    int8_t best_v = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > best_v) { best_v = data[i]; best_i = i; }
    }
    return best_i;
}

// ===================== Protocol =====================
// MCU -> PC: "REQ1" + seq(u32)
// PC  -> MCU: "KMN1" + seq(u32) + len(u32) + 784 bytes
// MCU -> PC: "RES1" + seq(u32) + pred(u8) + logits(10 int8)


static const uint8_t kMagicIn [4] = {'K','M','N','1'};
static const uint8_t kMagicOut[4] = {'R','E','S','1'};

// ===================== Main =====================
int main() {
    // İstersen tamamen kapat: debug print binary akışı bozabilir.
    // printf("\n=== KMNIST LeNet INT8 TFLM (REQ/RESP UART) ===\n");

    // 1) SDRAM init
    if (BSP_SDRAM_Init() != SDRAM_OK) {
        // printf("SDRAM init FAILED!\n");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // 2) SDRAM allocator reset
    g_sdram_offset = 0;

    // 3) Modeli SDRAM'e kopyala
    uint8_t* model_sdram = (uint8_t*)sdram_alloc((size_t)g_model_len, 16);
    if (!model_sdram) {
        while (true) { ThisThread::sleep_for(1000ms); }
    }
    memcpy(model_sdram, g_model, (size_t)g_model_len);

    // 4) Arena'yı SDRAM'den ayır
    uint8_t* tensor_arena = (uint8_t*)sdram_alloc(kTensorArenaSize, 16);
    if (!tensor_arena) {
        while (true) { ThisThread::sleep_for(1000ms); }
    }
    memset(tensor_arena, 0, kTensorArenaSize);

    // 5) TFLM setup
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel((const unsigned char*)model_sdram);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    static tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Input type kontrol (int8 bekliyoruz)
    const float in_scale = input->params.scale;
    const int   in_zero  = input->params.zero_point;

    // Buffers
    uint8_t hdr[12];
    uint8_t img[kImgSize];

    uint32_t seq = 1;

    while (true) {

    // 1️⃣ KMN1 magic ara
    bool found = false;
    uint8_t win[4] = {0};

    while (!found) {
        uint8_t b;
        if (g_serial.read(&b, 1) == 1) {
            memmove(win, win + 1, 3);
            win[3] = b;
            if (memcmp(win, kMagicIn, 4) == 0) {
                found = true;
            }
        } else {
            ThisThread::sleep_for(1ms);
        }
    }

    // 2️⃣ Header'ın geri kalanını oku
    // seq(u32) + len(u32)
    uint8_t hdr_rest[8];
    if (!read_exact(hdr_rest, 8)) continue;

    uint32_t seq = read_u32_le(&hdr_rest[0]);
    uint32_t len = read_u32_le(&hdr_rest[4]);

    if (len != kImgSize) {
        // yanlış payload → discard
        for (uint32_t i = 0; i < len; i++) {
            uint8_t dump;
            if (g_serial.read(&dump, 1) != 1) break;
        }
        continue;
    }

    // 3️⃣ Payload al
    uint8_t img[kImgSize];
    if (!read_exact(img, kImgSize)) continue;

    // 4️⃣ Input doldur
    if (input->type == kTfLiteInt8) {
        int8_t* in = input->data.int8;
        for (int i = 0; i < kImgSize; i++) {
            float x01 = img[i] / 255.0f;
            in[i] = quantize_01_to_int8(x01, in_scale, in_zero);
        }
    } else {
        memcpy(input->data.uint8, img, kImgSize);
    }

    // 5️⃣ Inference
    if (interpreter.Invoke() != kTfLiteOk) continue;

    // 6️⃣ Output hazırla
    uint8_t outbuf[4 + 4 + 1 + 10];
    memcpy(outbuf, kMagicOut, 4);
    write_u32_le(&outbuf[4], seq);

    int pred = 0;
    int8_t logits10[10];

    if (output->type == kTfLiteInt8) {
        const int8_t* out = output->data.int8;
        pred = argmax_int8(out, kNumClasses);
        memcpy(logits10, out, 10);
    }

    outbuf[8] = (uint8_t)pred;
    memcpy(&outbuf[9], logits10, 10);

    // 7️⃣ RES1 gönder
    write_all(outbuf, sizeof(outbuf));
}

}