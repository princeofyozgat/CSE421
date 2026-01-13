import os

def header_text(var_name: str) -> str:
    guard = f"_{var_name.upper()}_H_"
    return f"""#ifndef {guard}
#define {guard}

#include <cstdint>

extern const unsigned char {var_name}[];
extern const unsigned int {var_name}_len;

#endif
"""

def cc_text(data: bytes, var_name: str) -> str:
    lines = []
    lines.append(f'#include "{var_name}.h"\n')
    lines.append(f"alignas(16) const unsigned char {var_name}[] = {{\n")
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    lines.append("};\n")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};\n")
    return "".join(lines)

def main():
    tflite_path = os.path.join("tiny_efficientnet_mnist_int8.tflite")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError("tiny_efficientnet_mnist_int8.tflite yok. Önce convert scripti çalıştır.")

    # Mbed tarafına yaz (senin yapına göre burayı aynı kullandık)
    out_dir = os.path.join("..", "mbed", "models")
    os.makedirs(out_dir, exist_ok=True)

    var_name = "tiny_efficientnet_mnist_int8_model_data"

    with open(tflite_path, "rb") as f:
        data = f.read()

    h_path = os.path.join(out_dir, f"{var_name}.h")
    cc_path = os.path.join(out_dir, f"{var_name}.cc")

    with open(h_path, "w", encoding="utf-8") as f:
        f.write(header_text(var_name))
    with open(cc_path, "w", encoding="utf-8") as f:
        f.write(cc_text(data, var_name))

    print("Wrote:", h_path)
    print("Wrote:", cc_path)
    print("Bytes:", len(data))

if __name__ == "__main__":
    main()