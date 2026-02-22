import os
import re
import matplotlib.pyplot as plt

LOG_PATH = os.path.join("outputs", "logs", "pcam_full.log")

train_losses = []
val_losses   = []
val_accs     = []
val_aucs     = []

with open(LOG_PATH, "r") as f:
    for line in f:
        if "loss=" in line and "[Val" not in line:
            m = re.search(r"loss=([0-9.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))

        if "[Val" in line:
            m1 = re.search(r"loss=([0-9.]+)", line)
            m2 = re.search(r"acc=([0-9.]+)", line)
            m3 = re.search(r"auc=([0-9.]+)", line)
            if m1 and m2 and m3:
                val_losses.append(float(m1.group(1)))
                val_accs.append(float(m2.group(1)))
                val_aucs.append(float(m3.group(1)))

os.makedirs("outputs/plots", exist_ok=True)

plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("outputs/plots/train_loss.png")

plt.figure()
plt.plot(val_losses)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("outputs/plots/val_loss.png")

plt.figure()
plt.plot(val_accs)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("outputs/plots/val_accuracy.png")

plt.figure()
plt.plot(val_aucs)
plt.title("Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.savefig("outputs/plots/val_auc.png")

print("Plots saved in outputs/plots/")