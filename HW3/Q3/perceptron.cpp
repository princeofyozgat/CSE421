#include <math.h>
#include "perceptron.h"
#include "perceptron_weights.h"

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

int perceptron_predict(const float *x)
{
    float sum = perceptron_bias;

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        sum += perceptron_weights[i] * x[i];
    }

    float y = sigmoid(sum);

    return (y > 0.5f) ? 1 : 0;
}
