import numpy as np
from kmnist_io import load_kmnist

def write_model_header(path, array_name, data_bytes: bytes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"alignas(16) const unsigned char {array_name}[] = {{\n")
        for i, b in enumerate(data_bytes):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data_bytes)};\n")

def write_samples_header(path, images_u8, labels_u8):
    N = images_u8.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"static const int kNumSamples = {N};\n")
        f.write("static const int kImgH = 28;\nstatic const int kImgW = 28;\nstatic const int kImgC = 1;\n")
        f.write("static const int kImgSize = kImgH*kImgW*kImgC;\n\n")

        f.write("alignas(16) static const uint8_t kTestImages[kNumSamples][kImgSize] = {\n")
        for i in range(N):
            flat = images_u8[i].reshape(-1)  # 784
            f.write("  {")
            for j, v in enumerate(flat):
                if j % 32 == 0:
                    f.write("\n    ")
                f.write(f"{int(v)}, ")
            f.write("\n  },\n")
        f.write("};\n\n")

        f.write("static const uint8_t kTestLabels[kNumSamples] = {")
        for i in range(N):
            f.write(f"{int(labels_u8[i])}, ")
        f.write("};\n")

def main():
    with open("lenet5_kmnist_int8.tflite", "rb") as f:
        tflite_bytes = f.read()
    write_model_header("model_data.h", "g_model", tflite_bytes)

    _, (x_test, y_test) = load_kmnist()

    idx = [0, 1, 2, 3, 4]  # 5 sample
    images = x_test[idx].astype(np.uint8)  # (5,28,28,1) already uint8
    labels = y_test[idx].astype(np.uint8)

    write_samples_header("test_samples.h", images, labels)
    print("Wrote: model_data.h + test_samples.h")

if __name__ == "__main__":
    main()