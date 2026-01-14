import numpy as np

test_features = np.load("test_huMoments.npy")
test_labels = np.load("test_labels.npy")

NUM_SAMPLES = 5
NUM_FEATURES = 7

with open("mnist_test_data.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#define NUM_TEST_SAMPLES 5\n")
    f.write("#define NUM_FEATURES 7\n\n")

    f.write("const float test_features[NUM_TEST_SAMPLES][NUM_FEATURES] = {\n")
    for i in range(NUM_SAMPLES):
        row = ", ".join(f"{v:.6f}f" for v in test_features[i])
        f.write(f"  {{{row}}},\n")
    f.write("};\n\n")

    f.write("const int test_labels[NUM_TEST_SAMPLES] = {\n")
    for i in range(NUM_SAMPLES):
        f.write(f"  {test_labels[i]},\n")
    f.write("};\n")
