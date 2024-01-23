#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os
import h5py

plt.style.use("./presentation.mplstyle")

dataDir = "../../logs/test"

fig, ax = plt.subplots()
ax2=ax.twinx()
ax.set_zorder(-1)  # default zorder is 0 for ax1 and ax2
ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

lines = []
with h5py.File(os.path.join(dataDir, "data.hdf5"), "r", locking = False) as f:
    epochs = len(f["test"]["loss"][:])
    lines += ax.plot(np.arange(1, epochs+1), f["test"]["loss"][:], "-", color = "darkblue", label = r"$L^{\text{Test}}$")
    lines += ax.plot(np.arange(1, epochs+1), f["train"]["loss"][:], "-", color = "darkorange", label = r"$L^{\text{Tain}}$")
    lines += ax2.plot(np.arange(1, epochs+1), f["test"]["accuracy"][:], "--", color = "darkblue", label = r"Test Accuracy")
    lines += ax2.plot(np.arange(1, epochs+1), f["train"]["accuracy"][:], "--", color = "darkorange", label = r"Train Accuracy")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy", rotation = -90, labelpad = 10)
ax.set_xlim(1,epochs)
ax.set_ylim(0,None)
ax2.set_ylim(0,1)
ax.legend(lines, [line.get_label() for line in lines])
# plt.savefig(os.path.join(dataDir, f"training.pdf")) 
plt.show()
plt.close()