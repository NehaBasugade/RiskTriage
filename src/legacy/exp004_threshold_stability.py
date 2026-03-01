# src/experiments/exp004_threshold_stability.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

ALERT_RATE_URGENT = 0.35
ALERT_RATE_REVIEW = 0.60


def recall_at_rate(y_true, proba, rate):
    thr = np.quantile(proba, 1.0 - rate)
    pred = (proba >= thr).astype(int)
    return float(recall_score(y_true, pred, zero_division=0))


def curve(y_true, proba, rates):
    return np.array([recall_at_rate(y_true, proba, r) for r in rates])


def local_slope(y_vals, x_vals, x0):
    i = int(np.argmin(np.abs(x_vals - x0)))
    if i == 0:
        return float((y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0]))
    if i == len(x_vals) - 1:
        return float((y_vals[-1] - y_vals[-2]) / (x_vals[-1] - x_vals[-2]))
    return float((y_vals[i + 1] - y_vals[i - 1]) / (x_vals[i + 1] - x_vals[i - 1]))


def main():
    os.makedirs("runs/exp004", exist_ok=True)

    data = np.load("runs/exp004/probas_valid_calibrated.npz", allow_pickle=True)
    y = data["y_valid"]

    # Compare only the contenders we care about
    probas = {
        "RF_base": data["RF_base"],
        "RF_sigmoid": data["RF_sigmoid"],
        "RF_isotonic": data["RF_isotonic"],
    }

    rates = np.linspace(0.05, 0.90, 18)

    plt.figure()
    for name, p in probas.items():
        rec = curve(y, p, rates)
        plt.plot(rates, rec, marker="o", linestyle="-", label=name)

        if name in ("RF_base", "RF_sigmoid"):
            su = local_slope(rec, rates, ALERT_RATE_URGENT)
            sr = local_slope(rec, rates, ALERT_RATE_REVIEW)
            print(f"{name}: slope@0.35={su:.3f} slope@0.60={sr:.3f}")

    plt.axvline(ALERT_RATE_URGENT)
    plt.axvline(ALERT_RATE_REVIEW)
    plt.xlabel("Alert rate (fraction flagged)")
    plt.ylabel("Recall")
    plt.title("Recall vs Alert Rate (Validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = "runs/exp004/recall_vs_alert_rate.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()