#include <stdio.h>
#include "perceptron.h"

static const float test_zero_feature[26] = {
    22.401903f,     0.802237f,     -0.164720f,     0.460436f,     -0.988968f,
    -0.938986f,     -1.400651f,     0.343802f,     -0.196094f,     0.315614f,
    0.083614f,     0.325705f,     -0.052498f,     22.814875f,     2.208646f,
    -1.759750f,     -1.798975f,     -0.880139f,     0.453607f,     -2.143141f,
    0.209122f,     0.451033f,     0.659517f,     -0.696425f,     -0.032308f,
    0.090609f,
};


static const float test_one_feature[26] = {
    25.752718f,     0.229688f,     -0.295605f,     0.921813f,     -1.837648f,
    -2.110011f,     -0.943208f,     -0.466702f,     -0.778001f,     0.150827f,
    -0.228683f,     -0.230429f,     -0.823029f,     11.444330f,     4.594971f,
    1.244471f,     -1.414270f,     0.462675f,     -0.018820f,     -1.730232f,
    -1.095290f,     0.508063f,     0.446857f,     0.249186f,     -0.530569f,
    -0.070522f,
};


int main(void)
{
    int pred_zero = perceptron_predict(test_zero_feature);
    int pred_one  = perceptron_predict(test_one_feature);

    printf("Zero feature prediction     : %d (expected 0)\n", pred_zero);
    printf("Not-zero feature prediction : %d (expected 1)\n", pred_one);

    while (1);
    return 0;
}
