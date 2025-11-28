#ifndef KNN_CLS_INFERENCE_H
#define KNN_CLS_INFERENCE_H

#include "knn_mfcc_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void knn_cls_predict(const float *input, int *output);

int knn_cls_predict_label(const float *input);

#ifdef __cplusplus
}
#endif

#endif