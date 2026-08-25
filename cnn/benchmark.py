import pickle
import matplotlib.pyplot as plt
import os

# Load loss history
with open("cnn/model/history.pkl", "rb") as f:
    loss_history = pickle.load(f)

# Plot
plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/loss_curve_cnn.png")
plt.show()