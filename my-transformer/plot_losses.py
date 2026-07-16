"""
plot_losses.py — reads logs/loss_A.csv and logs/loss_B.csv and produces
loss_curve.png for the top of your README.

Usage:
    python plot_losses.py
"""
import csv
import matplotlib.pyplot as plt


def read_log(path):
    steps, train, val = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in [("A", "tab:blue"), ("B", "tab:orange")]:
        steps, train, val = read_log(f"logs/loss_{cfg}.csv")
        ax.plot(steps, train, color=color, linestyle="--", alpha=0.5, label=f"Model {cfg} train")
        ax.plot(steps, val, color=color, linestyle="-", label=f"Model {cfg} val")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Model A vs Model B — Training / Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=150)
    print("Saved loss_curve.png")


if __name__ == "__main__":
    main()
