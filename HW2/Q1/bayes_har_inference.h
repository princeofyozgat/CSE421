#ifndef BAYES_HAR_INFERENCE_H_INCLUDED
#define BAYES_HAR_INFERENCE_H_INCLUDED

#include "bayes_har_config.h"

#ifdef __cplusplus
extern "C" {
#endif


int bayes_har_predict(const float *features);

#ifdef __cplusplus
}
#endif

#endif  