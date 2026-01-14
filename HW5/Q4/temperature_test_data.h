#pragma once

#include <cstdint>

#define NUM_TEST_SAMPLES 5
#define NUM_FEATURES 5

const int8_t test_features[NUM_TEST_SAMPLES][NUM_FEATURES] = {
  {-48, -54, -58, -57, -46},
  {20, 11, 4, -4, -12},
  {51, 41, 34, 26, 20},
  {-24, -17, 1, 27, 54},
  {42, 34, 26, 20, 11},
};

const float test_labels[NUM_TEST_SAMPLES] = {
16.054, 17.077, 18.661, 22.373, 18.245
};
