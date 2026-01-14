#include <cstdio>
#include <cstdint>


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "temperature_model.h"
#include "temperature_test_data.h"

// Tensor arena size (gerekirse büyüt)
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
static tflite::MicroErrorReporter micro_error_reporter;


int main() {

    /*--------------------------------------------------
     * 1️⃣ Load model
     *--------------------------------------------------*/
    const tflite::Model* model =
        tflite::GetModel(temperature_model_int8_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return -1;
    }

    /*--------------------------------------------------
     * 2️⃣ Resolver + Interpreter
     *--------------------------------------------------*/
    tflite::AllOpsResolver resolver;

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize,&micro_error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed!\n");
        return -1;
    }

    /*--------------------------------------------------
     * 3️⃣ Get input / output tensors
     *--------------------------------------------------*/
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    printf("Input scale: %f zero_point: %d\n", input_scale, input_zero_point);
    printf("Output scale: %f zero_point: %d\n\n", output_scale, output_zero_point);

    /*--------------------------------------------------
     * 4️⃣ Inference loop
     *--------------------------------------------------*/
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {

        // Copy test feature vector to input tensor
        for (int j = 0; j < NUM_FEATURES; j++) {
            input->data.int8[j] = test_features[i][j];
        }

        // Run inference
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke failed at sample %d\n", i);
            continue;
        }

        // INT8 output
        int8_t y_q = output->data.int8[0];

        // Dequantize
        float y_pred =
            (y_q - output_zero_point) * output_scale;

        float y_true = test_labels[i];

        int y_true_int = (int)y_true;
        int y_true_frac = (int)((y_true - y_true_int) * 1000);
        if (y_true_frac < 0) y_true_frac = -y_true_frac;

        printf("Sample %d | True: %d.%03d | Predicted: %.3f\n",
            i,
            y_true_int,
            y_true_frac,
            y_pred
        );
    }

    return 0;
}
