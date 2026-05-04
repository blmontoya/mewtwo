import matplotlib.pyplot as plt
import numpy as np

iou1 = np.load("mious40.npy")
iou2 = np.load("mious.npy")

plt.plot(iou1, label="w/ log")
plt.plot(iou2, label="mean IoU")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mean IoU over Validation Set")
plt.legend()
plt.savefig("comparison_of_val.png")