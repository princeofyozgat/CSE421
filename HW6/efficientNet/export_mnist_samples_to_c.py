import os
import numpy as np

def to_c_array_u8(arr, name, cols=28):
    """
    arr: (N, 28, 28) uint8
    output: uint8_t name[N][784] = {...};
    """
    N = arr.shape[0]
    flat = arr.reshape(N, -1)

    lines = []
    lines.append("#pragma once\n")
    lines.append("#include <cstdint>\n\n")
    lines.append(f"static const int MNIST_SAMPLES = {N};\n")
    lines.append("static const int MNIST_W = 28;\n")
    lines.append("static const int MNIST_H = 28;\n\n")

    lines.append(f"static const uint8_t {name}[MNIST_SAMPLES][MNIST_W*MNIST_H] = {{\n")
    for i in range(N):
        lines.append("  {\n    ")
        for j in range(flat.shape[1]):
            lines.append(str(int(flat[i, j])))
            if j != flat.shape[1] - 1:
                lines.append(", ")
            # satır kır
            if (j + 1) % 32 == 0 and j != flat.shape[1] - 1:
                lines.append("\n    ")
        lines.append("\n  }")
        if i != N - 1:
            lines.append(",")
        lines.append("\n")
    lines.append("};\n\n")
    return "".join(lines)

def labels_to_c(labels, name="tiny_mnist_labels_5"):
    lines = []
    lines.append(f"static const uint8_t {name}[MNIST_SAMPLES] = {{ ")
    for i, v in enumerate(labels):
        lines.append(str(int(v)))
        if i != len(labels) - 1:
            lines.append(", ")
    lines.append(" };\n")
    return "".join(lines)

def main():
    
    mnist_npz = os.path.join("mnist.npz")
    if not os.path.exists(mnist_npz):
        raise FileNotFoundError(f"mnist.npz yok: {mnist_npz}")

    d = np.load(mnist_npz)
    x_test = d["x_test"]   # (10000, 28, 28) uint8
    y_test = d["y_test"]   # (10000,) uint8/int

    # ---- 5 örnek seçme yöntemi ----
    # A) Rastgele 5 örnek:
    # idx = np.random.choice(len(x_test), size=5, replace=False)

    # B) Her bir label'dan 1 örnek (0..4 gibi) seçelim (Daha iyi demo)
    wanted = [0, 1, 2, 3, 4]
    idx = []
    for w in wanted:
        pos = np.where(y_test == w)[0][0]
        idx.append(int(pos))
    idx = np.array(idx)

    samples = x_test[idx].astype(np.uint8)
    labels = y_test[idx].astype(np.uint8)


    out_path = os.path.join("tiny_mnist_samples_5.h")

    content = ""
    content += to_c_array_u8(samples, "tiny_mnist_images_5")
    content += labels_to_c(labels, "tiny_mnist_labels_5")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("OK -> yazıldı:", out_path)
    print("Seçilen label'lar:", labels.tolist())
    print("Seçilen index'ler:", idx.tolist())

if __name__ == "__main__":
    main()