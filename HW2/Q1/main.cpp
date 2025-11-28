#include "mbed.h"
#include "bayes_har_config.h"
#include "bayes_har_inference.h"

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "downstairs",
    "jogging",
    "sitting",
    "standing",
    "upstairs",
    "walking"
};

static const float example_features[NUM_FEATURES] = {
  1.1446199e+00, 9.8837757e+00 ,2.2133062e-02, 7.0000000e+01 ,8.0000000e+01,
 2.0000000e+01, 6.9450336e+00, 1.0195889e+01 ,7.0193176e+00 ,3.2658833e+01
};

int main()
{

    printf("Human Activity Recognition - Bayes Classifier (STM32F746G)\r\n");
    printf("NUM_CLASSES = %d, NUM_FEATURES = %d\r\n", NUM_CLASSES, NUM_FEATURES);


    int pred = bayes_har_predict(example_features);

    if (pred >= 0 && pred < NUM_CLASSES) {
        printf("Predicted class index: %d\r\n", pred);
        printf("Predicted activity  : %s\r\n", CLASS_NAMES[pred]);
    } else {
        printf("Prediction error! Invalid class index: %d\r\n", pred);
    }

    while (true) {
        ThisThread::sleep_for(1s);
    }
}