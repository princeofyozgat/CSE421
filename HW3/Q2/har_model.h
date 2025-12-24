#include <math.h>
#ifndef HAR_MODEL_H
#define HAR_MODEL_H

#define NUM_FEATURES 10

static const float har_weights[NUM_FEATURES] = {
    -0.1021077,  -0.27772403, -0.15186736, -0.00359923, -0.03850249,  0.04745133,
 -0.02765545,  0.05645446, -0.11725952,  0.03201351
};

static const float har_bias = 2.8729682;

static inline int har_predict(const float *x)
{
    float sum = har_bias;
    for (int i = 0; i < NUM_FEATURES; i++)
        sum += har_weights[i] * x[i];

    // sigmoid
    float y = 1.0f / (1.0f + expf(-sum));

    return (y > 0.5f) ? 1 : 0;
}

#endif
