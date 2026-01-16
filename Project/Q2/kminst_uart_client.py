import time
import struct
import numpy as np
import serial

PORT = "COM4"
BAUD = 115200

IMGS_NPY = "kmnist_test_imgs.npy"
LABELS_NPY = "kmnist_test_labels.npy"

START_INDEX = 0
COUNT = 5
SLEEP_BETWEEN = 0.05
MAGIC_REQ = b"REQ1" \
""
def recv_req(ser):
    find_magic(ser, MAGIC_REQ, timeout_s=5.0)
    rest = read_exact(ser, 4)
    seq = struct.unpack("<I", rest)[0]
    return seq

MAGIC_IN  = b"KMN1"
MAGIC_ACK = b"ACK1"
MAGIC_OUT = b"RES1"
IMG_SIZE = 28 * 28

def load_images(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 3 and x.shape[1:] == (28, 28):
        return x.astype(np.uint8)
    raise ValueError(f"Unexpected IMGS shape: {x.shape}")

def try_load_labels(path: str, n: int):
    try:
        y = np.load(path)
    except FileNotFoundError:
        return None
    y = np.asarray(y).reshape(-1)
    if y.shape[0] != n:
        raise ValueError("labels mismatch")
    return y.astype(np.int64)

def read_exact(ser, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise TimeoutError("timeout reading exact")
        data += chunk
    return data

def find_magic(ser, magic: bytes, timeout_s=2.0, max_skip=20000):
    t0 = time.time()
    win = b""
    skipped = 0
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timeout waiting for {magic!r}")
        b = ser.read(1)
        if not b:
            continue
        win = (win + b)[-len(magic):]
        if win == magic:
            return
        skipped += 1
        if skipped > max_skip:
            raise TimeoutError(f"Too much noise; never saw {magic!r}")

def send_frame(ser, seq: int, img_u8_flat: np.ndarray):
    img_u8_flat = img_u8_flat.astype(np.uint8).reshape(-1)
    payload = img_u8_flat.tobytes()
    header = MAGIC_IN + struct.pack("<II", seq, len(payload))
    ser.write(header + payload)
    ser.flush()

def recv_res1(ser):
    # after RES1: seq(u32) + pred(u8) + logits(10 int8)
    rest = read_exact(ser, 4 + 1 + 10)
    seq = struct.unpack("<I", rest[:4])[0]
    pred = rest[4]
    logits = np.frombuffer(rest[5:], dtype=np.int8, count=10)
    return seq, pred, logits

def main():
    x = load_images(IMGS_NPY)
    y = try_load_labels(LABELS_NPY, x.shape[0])

    ser = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(1.5)
    ser.reset_input_buffer()

    seq = 1
    idx = START_INDEX
    for _ in range(COUNT):
        img = x[idx]
    
        send_frame(ser, seq, img)
    
        find_magic(ser, MAGIC_OUT, timeout_s=5.0)
        rseq, pred, logits = recv_res1(ser)
    
        print(f"seq={rseq} pred={pred}")
    
        seq += 1
        idx += 1


    ser.close()

if __name__ == "__main__":
    main()