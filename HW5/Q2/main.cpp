#include "mbed.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kws_model.h"
#include "kws_test_data.h"

// =====================
// Tensor Arena
// =====================
#define ARENA_SIZE (80 * 1024)
static uint8_t tensor_arena[ARENA_SIZE];

// =====================
// Error Reporter
// =====================
static tflite::MicroErrorReporter micro_error_reporter;

int main()
{
    printf("KWS TFLite Micro Test\n");

    // =====================
    // Load model
    // =====================
    const tflite::Model* model =
        tflite::GetModel(kws_model_int8_tflite);

    // =====================
    // Resolver
    // =====================
    static tflite::AllOpsResolver resolver;

    // =====================
    // Interpreter (DOĞRU)
    // =====================
    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        ARENA_SIZE,
        &micro_error_reporter
    );

    // =====================
    // Allocate tensors
    // =====================
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        while (1);
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // =====================
    // Run test samples
    // =====================
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {

        // ⚠️ Quantize float → int8
        for (int j = 0; j < NUM_FEATURES; j++) {
            input->data.int8[j] =
                (int8_t)(test_features[i][j] / input->params.scale
                         + input->params.zero_point);
        }

        interpreter.Invoke();

        int pred = 0;
        int8_t max_val = output->data.int8[0];

        for (int k = 1; k < 10; k++) {
            if (output->data.int8[k] > max_val) {
                max_val = output->data.int8[k];
                pred = k;
            }
        }

        printf("Sample %d | True=%d | Pred=%d\n",
               i, test_labels[i], pred);
    }

    while (1);
}
