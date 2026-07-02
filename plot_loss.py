import os
import json
import matplotlib.pyplot as plt


# ==========================================
# Create plots folder
# ==========================================
os.makedirs("outputs/plots", exist_ok=True)

# ==========================================
# Load Loss History
# ==========================================
with open("outputs/loss_history.json", "r") as file:
    loss_history = json.load(file)

# ==========================================
# Plot Loss
# ==========================================
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.title("Training Loss vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

# ==========================================
# Save Figure
# ==========================================
plt.savefig(
    "outputs/plots/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Training loss graph saved to outputs/plots/training_loss.png")