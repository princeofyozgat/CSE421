#include "mbed.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>

// TFLM
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model + Samples
#include "resnetv1_mnist_int8_model_data.h"
#include "mnist_samples_5.h"

// UART
static BufferedSerial serial_port(USBTX, USBRX, 115200);

static void print_line(const char* s) {
    serial_port.write(s, strlen(s));
    serial_port.write("\r\n", 2);
}

static void print_i3(const char* prefix, int a, int b, int c) {
    char buf[96];
    snprintf(buf, sizeof(buf), "%s %d %d %d", prefix, a, b, c);
    print_line(buf);
}

// preprocess: 28->32, gray->RGB, float 0..1
static void preprocess_28_to_32_rgb(const uint8_t* in28, float* out32x32x3) {
    for (int y = 0; y < 32; y++) {
        int sy = (y * 28) / 32;
        for (int x = 0; x < 32; x++) {
            int sx = (x * 28) / 32;
            uint8_t pix = in28[sy * 28 + sx];
            float v = ((float)pix) / 255.0f;
            int idx = (y * 32 + x) * 3;
            out32x32x3[idx + 0] = v;
            out32x32x3[idx + 1] = v;
            out32x32x3[idx + 2] = v;
        }
    }
}

static void quantize_float_to_int8(const float* in_f, int8_t* out_i8, int len,
                                  float scale, int zero_point) {
    for (int i = 0; i < len; i++) {
        int q = (int)lrintf(in_f[i] / scale) + zero_point;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        out_i8[i] = (int8_t)q;
    }
}

static int argmax_int8(const int8_t* data, int len) {
    int best_i = 0;
    int best_v = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > best_v) { best_v = data[i]; best_i = i; }
    }
    return best_i;
}

int main() {
    print_line("=== ResNet v1 MNIST INT8 (TFLite Micro) ===");

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(resnetv1_mnist_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        print_line("Model schema mismatch!");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

        static tflite::MicroMutableOpResolver<14> resolver;
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddFullyConnected();
        resolver.AddAdd();
        resolver.AddConcatenation();
        resolver.AddMul();
        resolver.AddRelu();
        resolver.AddReshape();
        resolver.AddMaxPool2D();
        resolver.AddSoftmax();
        resolver.AddMean();          // GAP için
        resolver.AddQuantize();
        resolver.AddDequantize();
        resolver.AddLogistic();      // bazen activation graph’ta çıkar

    // Tensor arena (RAM)
    // AllocateTensors FAILED => bunu artır (RAM yetmezse board limitine takılabilirsin)
    static constexpr int kTensorArenaSize = 220 * 1024;
    static uint8_t tensor_arena[kTensorArenaSize];

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        print_line("AllocateTensors FAILED");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input->type != kTfLiteInt8) {
        print_line("Input not int8!");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // input dims kontrol
    if (input->dims->size != 4 ||
        input->dims->data[1] != 32 ||
        input->dims->data[2] != 32 ||
        input->dims->data[3] != 3) {
        print_line("Unexpected input shape!");
        print_i3("H W C:", input->dims->data[1], input->dims->data[2], input->dims->data[3]);
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    const float in_scale = input->params.scale;
    const int in_zero = input->params.zero_point;

    static float input_f[32 * 32 * 3];

    int correct = 0;

    for (int s = 0; s < MNIST_SAMPLES; s++) {
        preprocess_28_to_32_rgb(mnist_images_5[s], input_f);
        quantize_float_to_int8(input_f, input->data.int8, 32 * 32 * 3, in_scale, in_zero);

        if (interpreter.Invoke() != kTfLiteOk) {
            print_line("Invoke FAILED");
            while (true) { ThisThread::sleep_for(1000ms); }
        }

        const int8_t* out = output->data.int8;
        int pred = argmax_int8(out, 10);
        int true_label = (int)mnist_labels_5[s];

        if (pred == true_label) correct++;

        char buf[64];
        snprintf(buf, sizeof(buf), "Sample %d | True=%d Pred=%d", s, true_label, pred);
        print_line(buf);
    }

    char buf2[64];
    snprintf(buf2, sizeof(buf2), "Accuracy: %d/%d", correct, MNIST_SAMPLES);
    print_line(buf2);

    while (true) { ThisThread::sleep_for(1000ms); }
}