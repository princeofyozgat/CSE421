import re
import numpy as np
from joblib import load

MODEL_PATH = "temperature_prediction_lin.joblib"

# 1) PC model yükle (scikit-learn)
m = load(MODEL_PATH)

def pc_predict(x_list):
    """x_list: [t-5, t-4, t-3, t-2, t-1] float"""
    X = np.array(x_list, dtype=np.float32).reshape(1, -1)
    return float(m.predict(X)[0])

# 2) MCU logunu buraya yapıştır (senin çıktılar)
MCU_LOG = r"""
x=[22.100 22.300 22.400 22.500 22.600] -> y_hat=22.598
x=[22.300 22.400 22.500 22.600 22.598] -> y_hat=22.495
x=[22.400 22.500 22.600 22.598 22.495] -> y_hat=22.287
x=[22.500 22.600 22.598 22.495 22.287] -> y_hat=21.992
x=[22.600 22.598 22.495 22.287 21.992] -> y_hat=21.634
x=[22.598 22.495 22.287 21.992 21.634] -> y_hat=21.242
x=[22.495 22.287 21.992 21.634 21.242] -> y_hat=20.843
x=[22.287 21.992 21.634 21.242 20.843] -> y_hat=20.461
x=[21.992 21.634 21.242 20.843 20.461] -> y_hat=20.114
x=[21.634 21.242 20.843 20.461 20.114] -> y_hat=19.813
"""

# 3) Parse et
pattern = re.compile(
    r"x=\[(?P<x1>-?\d+\.\d+)\s+(?P<x2>-?\d+\.\d+)\s+(?P<x3>-?\d+\.\d+)\s+(?P<x4>-?\d+\.\d+)\s+(?P<x5>-?\d+\.\d+)\]\s*->\s*y_hat=(?P<y>-?\d+\.\d+)"
)

rows = []
for line in MCU_LOG.strip().splitlines():
    mth = pattern.search(line.strip())
    if not mth:
        continue
    x = [float(mth.group(f"x{i}")) for i in range(1, 6)]
    y_mcu = float(mth.group("y"))
    y_pc = pc_predict(x)
    rows.append((x, y_pc, y_mcu, y_pc - y_mcu))

# 4) Raporla
if not rows:
    print("Log parse edilemedi. Formatı kontrol et.")
else:
    print("idx | PC_pred   | MCU_pred  | diff(PC-MCU)")
    print("-----------------------------------------")
    for i, (_, y_pc, y_mcu, diff) in enumerate(rows):
        print(f"{i:>3} | {y_pc:>8.3f} | {y_mcu:>8.3f} | {diff:>+10.6f}")

    diffs = [abs(r[3]) for r in rows]
    print("\nMax abs diff:", max(diffs))
    print("Mean abs diff:", sum(diffs) / len(diffs))