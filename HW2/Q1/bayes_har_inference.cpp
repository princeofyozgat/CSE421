#include <math.h>
#include "bayes_har_inference.h"
#include "bayes_har_config.h"

/* Güvenlik için: math.h'da tanımlı değilse -INFINITY tanımla */
#ifndef INFINITY
#define INFINITY (1.0f/0.0f)
#endif

#ifndef M_LOG2PI
#define M_LOG2PI 1.8378770664093453f  /* log(2*pi) ≈ 1.837877 */
#endif

int bayes_har_predict(const float *features)
{
    float best_log_post = -INFINITY;
    int best_class = -1;

    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        /* diff = x - mean_c */
        float diff[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; ++i) {
            diff[i] = features[i] - MEANS[c][i];
        }

        /* tmp = INV_COVS[c] * diff */
        float tmp[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; ++i) {
            float s = 0.0f;
            for (int j = 0; j < NUM_FEATURES; ++j) {
                s += INV_COVS[c][i][j] * diff[j];
            }
            tmp[i] = s;
        }

        /* quad = diff^T * tmp */
        float quad = 0.0f;
        for (int i = 0; i < NUM_FEATURES; ++i) {
            quad += diff[i] * tmp[i];
        }

        /* log-likelihood ~ -0.5 * (quad + log(det) + D*log(2*pi)) */
        float log_det = logf(DETS[c]);
        float log_likelihood =
            -0.5f * (quad + log_det + NUM_FEATURES * M_LOG2PI);

        /* log-prior */
        float log_prior = logf(CLASS_PRIORS[c]);

        float log_post = log_prior + log_likelihood;

        if (log_post > best_log_post || best_class < 0) {
            best_log_post = log_post;
            best_class = c;
        }
    }

    return best_class;
}