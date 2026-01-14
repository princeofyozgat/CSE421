#include "mbed.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mnist_model.h"
#include "mnist_test_data.h"

#define ARENA_SIZE (60 * 1024)
static uint8_t tensor_arena[ARENA_SIZE];

static tflite::MicroErrorReporter error_reporter;

int main()
{
    printf("MNIST HuMoments TFLM\n");

    const tflite::Model* model =
        tflite::GetModel(mnist_hu_int8_tflite);

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter interpreter(
        model, resolver,
        tensor_arena, ARENA_SIZE,
        &error_reporter
    );

    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {

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
