// mbed/main.cpp
#include "mbed.h"

#include <cstdint>
#include <cstring>
#include <cmath>    // lrintf
#include <cstdio>   // snprintf

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model data
#include "squeezenet_mnist_int8_model_data.h"
#include "mnist_samples_5.h"

// =============== UART ===============
static BufferedSerial serial_port(USBTX, USBRX, 115200);

static void print_line(const char* s) {
    serial_port.write(s, strlen(s));
    serial_port.write("\r\n", 2);
}

static void print_fmt(const char* fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    print_line(buf);
}

// =============== Preprocess: 28x28 -> 32x32, 3 kanal ===============
static void preprocess_28_to_32_rgb(const uint8_t* in28, float* out32x32x3) {
    // nearest-neighbor scale: 28 -> 32
    for (int y = 0; y < 32; y++) {
        int sy = (y * 28) / 32;  // 0..27
        for (int x = 0; x < 32; x++) {
            int sx = (x * 28) / 32; // 0..27
            uint8_t pix = in28[sy * 28 + sx];
            float v = ((float)pix) / 255.0f; // 0..1
            int idx = (y * 32 + x) * 3;
            out32x32x3[idx + 0] = v;
            out32x32x3[idx + 1] = v;
            out32x32x3[idx + 2] = v;
        }
    }
}

static int argmax_int8(const int8_t* data, int len) {
    int best_i = 0;
    int best_v = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

static void quantize_input_float_to_int8(const float* in_f, int8_t* out_i8, int len,
                                        float scale, int zero_point) {
    for (int i = 0; i < len; i++) {
        int q = (int)lrintf(in_f[i] / scale) + zero_point;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        out_i8[i] = (int8_t)q;
    }
}

int main() {
    print_line("=== SqueezeNet MNIST INT8 (TFLite Micro) ===");

    // Error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Load model
    const tflite::Model* model = tflite::GetModel(squeezenet_mnist_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        print_fmt("Model schema version mismatch! model=%d expected=%d",
                  model->version(), TFLITE_SCHEMA_VERSION);
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // Ops resolver
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
    // Eğer AllocateTensors FAILED alırsan burayı artır (RAM yetmezse azaltman gerekebilir).
    static constexpr int kTensorArenaSize = 250 * 1024;
    static uint8_t tensor_arena[kTensorArenaSize];

    // Interpreter
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        print_line("AllocateTensors FAILED");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Input kontrol
    if (input->type != kTfLiteInt8) {
        print_line("Input tensor is not int8. (Converter settings?)");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // Output kontrol
    if (output->type != kTfLiteInt8) {
        print_line("Output tensor is not int8.");
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // Input shape kontrol (beklenen: [1,32,32,3])
    // Bazı TFLM sürümlerinde dims->size/dims->data vardır.
    if (input->dims->size != 4 ||
        input->dims->data[1] != 32 ||
        input->dims->data[2] != 32 ||
        input->dims->data[3] != 3) {
        print_fmt("Unexpected input shape: [%d,%d,%d,%d]",
                  input->dims->data[0], input->dims->data[1],
                  input->dims->data[2], input->dims->data[3]);
        while (true) { ThisThread::sleep_for(1000ms); }
    }

    // Quant params
    const float in_scale = input->params.scale;
    const int in_zero = input->params.zero_point;

    print_fmt("Input quant: scale=%f zero=%d", in_scale, in_zero);
    print_fmt("Running %d MNIST samples...", MNIST_SAMPLES);

    // Preprocess buffer
    static float input_f[32 * 32 * 3];

    int correct = 0;

    for (int s = 0; s < MNIST_SAMPLES; s++) {
        // 1) preprocess
        preprocess_28_to_32_rgb(mnist_images_5[s], input_f);

        // 2) quantize into model input tensor
        int8_t* input_i8 = input->data.int8;
        quantize_input_float_to_int8(input_f, input_i8, 32 * 32 * 3, in_scale, in_zero);

        // 3) invoke
        if (interpreter.Invoke() != kTfLiteOk) {
            print_fmt("Invoke FAILED at sample %d", s);
            while (true) { ThisThread::sleep_for(1000ms); }
        }

        // 4) read output & argmax
        const int8_t* out = output->data.int8;
        int pred = argmax_int8(out, 10);
        int true_label = (int)mnist_labels_5[s];

        if (pred == true_label) correct++;

        print_fmt("Sample %d | True=%d Pred=%d", s, true_label, pred);

        // İstersen her sample için raw output bas:
        // print_line("Raw out int8:");
        // for (int i = 0; i < 10; i++) { print_fmt("  [%d]=%d", i, (int)out[i]); }
    }

    print_fmt("Accuracy on 5 samples: %d/%d", correct, MNIST_SAMPLES);

    while (true) {
        ThisThread::sleep_for(1000ms);
    }
}