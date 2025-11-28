#include "knn_cls_inference.h"
#include <float.h>
#include <math.h>

void knn_cls_predict(const float *input, int *output)
{
    float best_dist[NUM_NEIGHBORS];
    int   best_label[NUM_NEIGHBORS];

    for (int k = 0; k < NUM_NEIGHBORS; ++k)
    {
        best_dist[k]  = FLT_MAX;
        best_label[k] = -1;
    }

    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        float dist = 0.0f;
        for (int j = 0; j < NUM_FEATURES; ++j)
        {
            float diff = input[j] - DATA[i][j];
            dist += diff * diff;
        }

        int insert_pos = -1;

        if (dist < best_dist[NUM_NEIGHBORS - 1])
        {
            insert_pos = NUM_NEIGHBORS - 1;

            while (insert_pos > 0 && dist < best_dist[insert_pos - 1])
            {
                best_dist[insert_pos]  = best_dist[insert_pos - 1];
                best_label[insert_pos] = best_label[insert_pos - 1];
                insert_pos--;
            }

            best_dist[insert_pos]  = dist;
            best_label[insert_pos] = DATA_LABELS[i];
        }
    }

    for (int c = 0; c < NUM_CLASSES; ++c)
        output[c] = 0;

    for (int k = 0; k < NUM_NEIGHBORS; ++k)
    {
        int lbl = best_label[k];
        if (lbl >= 0 && lbl < NUM_CLASSES)
            output[lbl]++;
    }
}


int knn_cls_predict_label(const float *input)
{
    int votes[NUM_CLASSES];

    knn_cls_predict(input, votes);

    int best_class = 0;
    int best_vote  = votes[0];

    for (int c = 1; c < NUM_CLASSES; ++c)
    {
        if (votes[c] > best_vote)
        {
            best_vote  = votes[c];
            best_class = c;
        }
    }

    return best_class;
}