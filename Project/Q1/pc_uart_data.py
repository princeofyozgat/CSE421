import json
import struct
import time
import numpy as np
import serial
import tensorflow as tf

# -----------------------------
# DATA: MNIST digit -> 96x96 canvas sample
# -----------------------------
def make_canvas_sample(canvas=96, digit_class=7, seed=123):
    rng = np.random.default_rng(seed)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    idxs = np.where(y_train == digit_class)[0]
    idx = int(rng.choice(idxs))
    digit = x_train[idx].astype(np.uint8)  # 28x28

    target = 34
    digit_tf = tf.image.resize(digit[..., None], (target, target), method="bilinear")
    digit_tf = tf.clip_by_value(digit_tf, 0.0, 255.0)
    digit_rs = tf.cast(digit_tf, tf.uint8).numpy()  # (t,t,1)

    y0, x0 = 20, 30
    canvas_img = np.zeros((canvas, canvas, 1), dtype=np.uint8)
    canvas_img[y0:y0+target, x0:x0+target, :] = np.maximum(
        canvas_img[y0:y0+target, x0:x0+target, :],
        digit_rs
    )

    return canvas_img[:, :, 0]  # (96,96) uint8


# -----------------------------
# UART protocol: header + payload
# -----------------------------
def send_image(ser, img_u8):
    H, W = img_u8.shape
    C = 1
    payload = img_u8.tobytes()
    hdr = struct.pack("<2sHHBBI", b"IM", H, W, C, 0, len(payload))  # dtype=0(u8)
    ser.write(hdr + payload)
    ser.flush()


def read_packet(ser, timeout_s=10.0):
    """
    UART'tan satır satır okur.
    @LOG  -> ekrana basar, devam eder
    @READY/@RES/@ERR -> JSON parse eder, return eder
    timeout olursa None döner
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        raw = ser.readline()
        if not raw:
            continue

        s = raw.decode("utf-8", errors="ignore").strip()
        if not s:
            continue

        if s.startswith("@LOG "):
            print(s[5:])
            continue

        if s.startswith("@READY "):
            payload = s[7:].strip()
            try:
                return ("READY", json.loads(payload))
            except Exception:
                return ("READY", {"raw": payload})

        if s.startswith("@RES "):
            payload = s[5:].strip()
            try:
                return ("RES", json.loads(payload))
            except Exception:
                return ("RES", {"raw": payload})

        if s.startswith("@ERR "):
            payload = s[5:].strip()
            try:
                return ("ERR", json.loads(payload))
            except Exception:
                return ("ERR", {"raw": payload})

        print("UNKNOWN:", s)

    return None


def fixed_to_float(i_part, f_part, decimals=6):
    return float(i_part) + float(f_part) / (10 ** decimals)


def main():
    COM = "COM4"     # <-- kendi COM’un
    BAUD = 115200

    # 1) Serial aç (timeout küçük olacak, read_packet kendi timeoutunu yönetiyor)
    ser = serial.Serial(COM, BAUD, timeout=0.2)

    # 2) Port açılır açılmaz buffer temizle (ESKİ JUNK SILINSIN)
    ser.reset_input_buffer()

    # 3) Kart reset atıp boot mesajlarını basması için biraz bekle
    time.sleep(2.0)

    # 4) READY gelene kadar bekle (MCU artık periyodik READY basacak)
    pkt = read_packet(ser, timeout_s=15.0)
    print("MCU READY:", pkt)

    if pkt is None or pkt[0] != "READY":
        print("READY gelmedi. COM doğru mu? Kart çalışıyor mu? timeout'u 30 yapıp dene.")
        ser.close()
        return

    digit_list = [0,1,2,3,4,5,6,7,8,9]
    n_samples = 10

    for i in range(n_samples):
        d = digit_list[i % len(digit_list)]
        seed = 123 + i

        print(f"\n=== SEND #{i} digit={d} seed={seed} ===")
        img = make_canvas_sample(canvas=96, digit_class=d, seed=seed)

        # Gönder
        send_image(ser, img)

        # Cevap bekle
        # ---- BURASI: RES/ERR gelene kadar bekle (READY gelirse yut) ----
        t0 = time.time()
        resp = None
        kind = None
        
        while time.time() - t0 < 15.0:
            pkt = read_packet(ser, timeout_s=2.0)  # kısa kısa oku
            if pkt is None:
                continue
            
            kind, resp = pkt
        
            # Bazı anlarda READY tekrar düşebilir -> ignore
            if kind == "READY":
                continue
            
            # Beklediğimiz şeyler bunlar:
            if kind in ("RES", "ERR"):
                break
            
        if kind is None or resp is None or kind == "READY":
            print("RESP RAW: None (timeout)")
            continue
        
        print("RESP RAW:", resp)
        
        if kind == "ERR":
            print("MCU ERROR:", resp)
            continue

        if kind == "RES" and isinstance(resp, dict) and resp.get("ok") is True and "score_i" in resp:
            score = fixed_to_float(resp["score_i"], resp["score_f"], 6)
            bi = resp["box_i"]
            bf = resp["box_f"]
            ymin = fixed_to_float(bi[0], bf[0], 6)
            xmin = fixed_to_float(bi[1], bf[1], 6)
            ymax = fixed_to_float(bi[2], bf[2], 6)
            xmax = fixed_to_float(bi[3], bf[3], 6)

            print(f"PARSED: cls={resp['cls']} score={score:.6f} "
                  f"box=[{ymin:.6f},{xmin:.6f},{ymax:.6f},{xmax:.6f}]")

        if kind == "ERR":
            print("MCU ERROR:", resp)

    ser.close()


if __name__ == "__main__":
    main()