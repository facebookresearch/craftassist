from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import visdom
import pickle
import os

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
MC_DIR = os.path.join(GEOSCORER_DIR, "../../../")

A = Axes3D  # to make flake happy :(


def cuboid_data(pos, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]
    y = [
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1], o[1], o[1]],
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]
    z = [
        [o[2], o[2], o[2], o[2], o[2]],
        [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), color=(0, 1, 0, 1), ax=None):
    # Plotting a cube element at position pos
    if ax is not None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)


class SchematicPlotter:
    def __init__(self, viz):
        self.viz = viz
        ims = pickle.load(
            open(os.path.join(MC_DIR, "minecraft_specs/block_images/block_data"), "rb")
        )
        colors = []
        alpha = []
        self.bid_to_color = {}
        for b, I in ims["bid_to_image"].items():
            I = I.reshape(1024, 4)
            if all(I[:, 3] < 0.2):
                colors = (0, 0, 0)
            else:
                colors = I[I[:, 3] > 0.2, :3].mean(axis=0) / 256.0
            alpha = I[:, 3].mean() / 256.0
            self.bid_to_color[b] = (colors[0], colors[1], colors[2], alpha)

    def draw(self, schematic, n=1, title=""):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_aspect("equal")
        if type(schematic) is np.ndarray:
            for i in range(schematic.shape[0]):
                for j in range(schematic.shape[1]):
                    for k in range(schematic.shape[2]):
                        if schematic[i, j, k, 0] > 0:
                            c = self.bid_to_color.get(tuple(schematic[i, j, k, :]))
                            if c:
                                plotCubeAt(pos=(i, k, j), color=c, ax=ax)  # x, z, y
        else:
            for b in schematic:
                if b[1][0] > 0:
                    c = self.bid_to_color.get(b[1])
                    if c:
                        plotCubeAt(pos=(b[0][0], b[0][2], b[0][1]), color=c, ax=ax)  # x, z, y
        plt.title(title)
        visrotate(n, ax, self.viz)

        return fig, ax


def visrotate(n, ax, viz):
    for angle in range(45, 405, 360 // n):
        ax.view_init(30, angle)
        plt.draw()
        viz.matplot(plt)


if __name__ == "__main__":
    import sys

    CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
    sys.path.append(CRAFTASSIST_DIR)
    import shapes

    viz = visdom.Visdom(server="http://localhost")
    sp = SchematicPlotter(viz)
    #    schematic = np.load('/private/home/aszlam/tmpmc/ts.npy')
    schematic = shapes.square_pyramid()
    fig, ax = sp.draw(schematic, 4, "yo")
