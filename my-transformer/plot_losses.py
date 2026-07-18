"""Create a loss convergence plot for Model A and Model B."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    logs_dir = Path("logs")
    a_path = logs_dir / "loss_A.csv"
    b_path = logs_dir / "loss_B.csv"

    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Expected logs/loss_A.csv and logs/loss_B.csv before plotting.")

    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    plt.figure(figsize=(9, 5))
    plt.plot(a["step"], a["val_loss"], label="Model A Validation Loss")
    plt.plot(b["step"], b["val_loss"], label="Model B Validation Loss")
    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.title("Loss Convergence: Model A vs Model B")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    print("Saved loss_curve.png")


if __name__ == "__main__":
    main()
