#include "mbed.h"
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "har_model_data.h"
#include "test_inputs.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


extern const unsigned char mlp_har_int8_tflite[];
extern const unsigned int  mlp_har_int8_tflite_len;

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

constexpr int kTensorArenaSize = 48 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

static int argmax_int8(const int8_t* data, int n) {
    int best_i = 0;
    int best_v = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

int main() {
    printf("\n=== HAR 11.6 (INT8) Inference Start ===\n");

    const tflite::Model* model = tflite::GetModel(mlp_har_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch: model=%d expected=%d\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors FAILED. Arena size yetmiyor olabilir.\n");
        return 1;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Input type=%d dims=%d shape=", input->type, input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("\n");

    float in_scale = input->params.scale;
    int   in_zp    = input->params.zero_point;
    printf("Input quant: scale=%f zero_point=%d\n", in_scale, in_zp);

    for (int s = 0; s < kNumSamples; s++) {

        for (int i = 0; i < kNumFeatures; i++) {
            float x = g_test_inputs[s][i];

            int32_t q = (int32_t) lroundf(x / in_scale) + in_zp;

            if (q > 127)  q = 127;
            if (q < -128) q = -128;

            input->data.int8[i] = (int8_t)q;
        }

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke FAILED at sample=%d\n", s);
            continue;
        }

        int pred = argmax_int8(output->data.int8, 6);

        printf("sample=%02d  pred=%d  true=%d\n", s, pred, g_test_labels[s]);
    }

    printf("=== Done ===\n");
    while (true) {
        ThisThread::sleep_for(1000ms);
    }
}