#include <cstdint>
#include <cstdio>
#include <cmath>

// TFLM
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mobilenet_mnist_int8_model_data.h"
#include "mnist_samples_5.h"

constexpr int kTensorArenaSize = 120 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
static float bilinear_at(
    const uint8_t* img, int w, int h, float x, float y)
{
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x1 >= w) x1 = w - 1;
    if (y1 >= h) y1 = h - 1;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * w + x0];
    float v10 = img[y0 * w + x1];
    float v01 = img[y1 * w + x0];
    float v11 = img[y1 * w + x1];

    float v0 = v00 + dx * (v10 - v00);
    float v1 = v01 + dx * (v11 - v01);
    return v0 + dy * (v1 - v0);
}
int main() {
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model =
        tflite::GetModel(mobilenet_mnist_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema mismatch!");
        return -1;
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

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("AllocateTensors failed");
        return -1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    const float input_scale = input->params.scale;
    const int input_zero = input->params.zero_point;

    const float output_scale = output->params.scale;
    const int output_zero = output->params.zero_point;

    for (int s = 0; s < MNIST_SAMPLES; s++) {
        // ---- Input: 28x28 → 32x32x3 ----
        int idx = 0;
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                uint8_t pix = 0;
                int idx = 0;

for (int oy = 0; oy < 32; oy++) {
    for (int ox = 0; ox < 32; ox++) {

        // 32 -> 28 mapping
        float src_x = ox * (27.0f / 31.0f);
        float src_y = oy * (27.0f / 31.0f);

        float pix = bilinear_at(
            mnist_images_5[s],
            28, 28,
            src_x, src_y
        );

        // normalize
        float norm = pix / 255.0f;

        // quantize (INT8)
        int32_t q = (int32_t)round(norm / input_scale) + input_zero;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;

        int8_t q8 = (int8_t)q;

        // RGB (3 kanal aynı)
        input->data.int8[idx++] = q8;
        input->data.int8[idx++] = q8;
        input->data.int8[idx++] = q8;
    }
}


                float norm = pix / 255.0f;
                int8_t q = (int8_t)round(norm / input_scale) + input_zero;

                for (int c = 0; c < 3; c++)
                    input->data.int8[idx++] = q;
            }
        }

        interpreter.Invoke();

        // ---- Argmax ----
        int best = 0;
        int8_t best_val = output->data.int8[0];
        for (int i = 1; i < 10; i++) {
            if (output->data.int8[i] > best_val) {
                best_val = output->data.int8[i];
                best = i;
            }
        }

        printf("Sample %d | GT=%d | Pred=%d\n",
               s, mnist_labels_5[s], best);
    }

    return 0;
}
