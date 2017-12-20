""" Generate plots for the report
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from saliency import saliency, data, model

def sequential():
    sns.set_context("paper", font_scale=2.5)

    X, y = data.load_images("data/imgs")
    ittykoch = model.IttyKoch(n_jobs=4)
    S = ittykoch.predict(X[:-1])

    for i in range(len(S)):
        salmap = saliency.attenuate_borders(S[i][...,np.newaxis], 5)[...,0]

        salmap = np.exp(salmap)
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())

        fix, final = saliency.sequential_salicency(salmap, 8, 10)

        fig, axes = plt.subplots(1,3,figsize=(20,8))

        axes[2].imshow(saliency.resize(X[i], 64))
        axes[2].plot(fix[:,1], fix[:,0], c="blue", linewidth=4, markersize=20)
        axes[2].scatter(fix[:,1], fix[:,0], marker="o", cmap="Blues", c=np.arange(len(fix)), s=300)

        axes[1].imshow(final)
        axes[0].imshow(S[i])

        for ax, title in zip(axes, ['Saliency', 'Attenuated', 'Gaze']):
            ax.axis("off")
            ax.set_title(title)

        plt.tight_layout()
        plt.savefig("report/fig/sequence-{}.pdf".format(i), bbox_inches="tight")

def ittykoch():
    sns.set_context("paper", font_scale=2.5)

    X, y = data.load_images("data/imgs")
    ittykoch = model.IttyKoch(n_jobs=4)
    S = ittykoch.predict(X[:-1])

    sns.set_context("paper", font_scale=2)

    fig, axes = plt.subplots(4,6,figsize=(20,10))
    for i, (ax_img, ax_sal, ax_logsal, ax_gt) in enumerate(zip(*axes)):
        ax_img.imshow(X[i])
        ax_sal.imshow(S[i])
        ax_logsal.imshow(-np.log(1+1e-5-S[i]))
        if y[i] is not None:
            ax_gt.imshow(y[i])
        else:
            ax_gt.axis("off")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid("off")

    for ax, t in zip(axes[:,0], ["Input", "S", "log S", "Ground Truth"]):
        ax.set_ylabel(t)

    plt.tight_layout()
    plt.savefig("report/fig/output.pdf", bbox_inches="tight")

def gabor():
    sns.set_context("paper", font_scale=2.5)
    fig, axes = plt.subplots(3,5,figsize=(20,10))

    for i, ax_ in enumerate(axes):
        for j, ax in enumerate(ax_):
            angle = np.pi*j/axes.shape[1]
            wavelen = 3 + i*6./axes.shape[0]

            gab = saliency.construct_gabor(angle, phase=np.pi/2, wavelen=wavelen)
            ax.imshow(gab, cmap="gray")
            ax.set_title("$\\theta={:.1f} \\pi, \\lambda={:.1f}$".format(angle/np.pi, wavelen))
            ax.axis("off")

    plt.savefig("report/fig/gabor.pdf", bbox_inches="tight")

if __name__ == '__main__':
    gabor()
    ittykoch()
    sequential()
