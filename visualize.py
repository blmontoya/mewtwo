import matplotlib.pyplot as plt
import numpy as np

losses = np.load("train_loss.npy")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss.png")