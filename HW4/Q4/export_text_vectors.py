import numpy as np

# INT8 test vektörlerini yükle (önceden quantize edilmiş!)
X_q = np.load("test_features_int8.npy")   # <-- ÇOK KRİTİK
y = np.load("test_labels_5.npy")

with open("temperature_test_data.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <cstdint>\n\n")

    f.write("#define NUM_TEST_SAMPLES 5\n")
    f.write("#define NUM_FEATURES 5\n\n")

    # Features
    f.write("const int8_t test_features[NUM_TEST_SAMPLES][NUM_FEATURES] = {\n")
    for sample in X_q:
        row = ", ".join(str(int(v)) for v in sample)
        f.write(f"  {{{row}}},\n")
    f.write("};\n\n")

    # Labels (float olabilir, regression!)
    f.write("const float test_labels[NUM_TEST_SAMPLES] = {\n")
    f.write(", ".join(f"{float(v):.3f}" for v in y))
    f.write("\n};\n")

print("✔ INT8 test vectors exported to temperature_test_data.h")
