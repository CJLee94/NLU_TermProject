import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# plt.ion()

out_dir = "albert-mnli-train-True-True"
total_epoch = 7

aum = []

for i in range(total_epoch):
    aum.append(torch.load(os.path.join(out_dir, "aum_{}.pt".format(i))).detach().cpu().numpy())

aum = np.array(aum)

with open(os.path.join(out_dir, "flip_index.json"), "r") as fp:
    flip_index = json.load(fp)

correct = np.ones(aum.shape[1], np.bool)
correct[flip_index] = 0

aum_mis = aum[:, flip_index]
aum_cor = aum[:, correct]

fig1, ax1 = plt.subplots()

ax1.plot(aum.mean(axis=1))
ax1.plot(aum_mis.mean(axis=1))
ax1.plot(aum_cor.mean(axis=1))

fig2, ax2 = plt.subplots()
bins = np.linspace(-5,3, 500)
freq_cor, bins_cor = np.histogram(aum_cor[0], bins=bins, density=True)
freq_mis, bins_mis = np.histogram(aum_mis[0], bins=bins, density=True)
bar_cor = ax2.bar(bins_cor[:-1], freq_cor, 0.05, color="green", align="edge")
bar_mis = ax2.bar(bins_mis[:-1], -freq_mis, 0.05, color="orange", align="edge")
ax2.set_ylim([-2, 2])
ax2.margins(x=0)
plt.subplots_adjust(bottom=0.25)
axepoch = plt.axes([0.125, 0.1, 0.78, 0.03])
# # freq, scalar = np.hist(aum[0], bins=100, density=True)
epochslider = Slider(ax=axepoch,
                     label="Epoch",
                     valmin=0,
                     valmax=total_epoch,
                     valinit=0)

# def renew_freq()

def update(val):
    # val = int(val)
    index = int(epochslider.val)
    bins = np.linspace(-5, 3, 500)
    freq_cor, bins_cor = np.histogram(aum_cor[index], bins=bins, density=True)
    freq_mis, bins_mis = np.histogram(aum_mis[index], bins=bins, density=True)
    for idx, (rec_cor, rec_mis) in enumerate(zip(bar_cor, bar_mis)):
        rec_cor.set_height(freq_cor[idx])
        rec_mis.set_height(-freq_mis[idx])
    fig2.canvas.draw_idle()
#
epochslider.on_changed(update)
# update(0)

plt.show()