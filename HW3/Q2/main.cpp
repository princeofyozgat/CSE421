#include "mbed.h"
#include "har_model.h"

int main()
{
    float notwalking_features[NUM_FEATURES] = {
        1.57059635,  9.74859363,  0.36025818, 59.0, 78.0, 32.0,
        10.94113081, 22.83305572, 12.94758898, 66.32483691
    };
    float walking_features[NUM_FEATURES] = {
        1.14461997e+00, 9.88377563e+00, 2.21330619e-02, 7.00000000e+01,
        8.00000000e+01, 2.00000000e+01, 6.94503336e+00, 1.01958899e+01,
        7.01931749e+00, 3.26588326e+01
    };

    printf("Walking example: %s\n", har_predict(walking_features) == 0 ? "Walking" : "Not Walking");

    printf("Not-Walking example: %s\n",har_predict(notwalking_features) == 0 ? "Walking" : "Not Walking");

    while (true) {}
}
