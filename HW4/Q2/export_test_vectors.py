import numpy as np

# === Load existing test vectors ===
test_features = np.load("test_features_5.npy")
test_labels   = np.load("test_labels_5.npy")     

assert test_features.shape == (5, 26)
assert test_labels.shape == (5,)

# === Export to C header ===
with open("kws_test_data.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write("#define NUM_TEST_SAMPLES 5\n")
    f.write("#define NUM_FEATURES 26\n\n")

    # Features
    f.write("const float test_features[NUM_TEST_SAMPLES][NUM_FEATURES] = {\n")
    for vec in test_features:
        f.write("  {")
        f.write(", ".join(f"{x:.6f}f" for x in vec))
        f.write("},\n")
    f.write("};\n\n")

    # Labels
    f.write("const int test_labels[NUM_TEST_SAMPLES] = { ")
    f.write(", ".join(str(int(x)) for x in test_labels))
    f.write(" };\n")

print("kws_test_data.h generated successfully")
