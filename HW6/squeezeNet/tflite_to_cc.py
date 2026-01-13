# train/tflite_to_cc.py
import os

def bytes_to_c_array(data: bytes, var_name: str) -> str:
    lines = []
    lines.append(f'#include "{var_name}.h"\n')
    lines.append(f'alignas(16) const unsigned char {var_name}[] = {{\n')

    # 12 byte per line
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {line},\n")

    lines.append("};\n")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};\n")
    return "".join(lines)

def header_text(var_name: str) -> str:
    guard = f"_{var_name.upper()}_H_"
    return f"""#ifndef {guard}
#define {guard}

#include <cstdint>

extern const unsigned char {var_name}[];
extern const unsigned int {var_name}_len;

#endif
"""

def main():
    

    tflite_path = os.path.join("squeezenet_mnist_int8.tflite")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError("Önce convert_to_tflite.py çalıştır. .tflite yok.")


    var_name = "squeezenet_mnist_int8_model_data"

    with open(tflite_path, "rb") as f:
        data = f.read()

    cc_path = os.path.join(f"{var_name}.cc")
    h_path  = os.path.join(f"{var_name}.h")

    with open(cc_path, "w", encoding="utf-8") as f:
        f.write(bytes_to_c_array(data, var_name))

    with open(h_path, "w", encoding="utf-8") as f:
        f.write(header_text(var_name))

    print("Wrote:", cc_path)
    print("Wrote:", h_path)
    print("Bytes:", len(data))

if __name__ == "__main__":
    main()