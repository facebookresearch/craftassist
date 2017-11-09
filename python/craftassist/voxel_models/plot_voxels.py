"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import plotly.graph_objs as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import visdom
import pickle
import os
import torch

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

    def drawMatplot(self, schematic, n=1, title=""):
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

    def drawPlotly(self, schematic):
        x = []
        y = []
        z = []
        id = []
        if type(schematic) is torch.Tensor:
            sizes = list(schematic.size())
            for i in range(sizes[0]):
                for j in range(sizes[1]):
                    for k in range(sizes[2]):
                        if schematic[i, j, k] > 0:
                            x.append(i)
                            y.append(j)
                            z.append(k)
                            id.append(schematic[i, j, k].item())
        elif type(schematic) is np.ndarray:
            for i in range(schematic.shape[0]):
                for j in range(schematic.shape[1]):
                    for k in range(schematic.shape[2]):
                        if schematic[i, j, k, 0] > 0:
                            c = self.bid_to_color.get(tuple(schematic[i, j, k, :]))
                            print(tuple(schematic[i, j, k, :]))
                            if c:
                                x.append(i)
                                y.append(j)
                                z.append(k)
                                id.append(i + j + k)
        else:
            for b in schematic:
                if b[1][0] > 0:
                    c = self.bid_to_color.get(b[1])
                    if c:
                        x.append(b[0][0])
                        y.append(b[0][2])
                        z.append(b[0][1])
                        id.append(i + j + k)

        trace1 = go.Scatter3d(
            x=np.asarray(x).transpose(),
            y=np.asarray(y).transpose(),
            z=np.asarray(z).transpose(),
            mode="markers",
            marker=dict(
                size=5,
                symbol="square",
                color=id,
                colorscale="Viridis",
                line=dict(color="rgba(217, 217, 217, 1.0)", width=0),
                opacity=1.0,
            ),
        )
        data = [trace1]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        self.viz.plotlyplot(fig)
        return fig


def visrotate(n, ax, viz):
    for angle in range(45, 405, 360 // n):
        ax.view_init(30, angle)
        plt.draw()
        viz.matplot(plt)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="shapes",
        help="which\
                        dataset to visualize (shapes|segments)",
    )
    opts = parser.parse_args()

    CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
    sys.path.append(CRAFTASSIST_DIR)

    vis = visdom.Visdom(server="http://localhost")
    sp = SchematicPlotter(vis)
    # fig, ax = sp.drawMatplot(schematic, 4, "yo")

    if opts.dataset == "shapes":
        import shape_dataset as sdata

        num_examples = 3
        num_neg = 3
        dataset = sdata.SegmentCenterShapeData(
            nexamples=num_examples, for_vis=True, useid=True, shift_max=10, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
    elif opts.dataset == "segments":
        import inst_seg_dataset as idata

        num_examples = 1
        num_neg = 1
        dataset = idata.SegmentCenterInstanceData(
            nexamples=num_examples, shift_max=10, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
    else:
        raise Exception("Unknown dataset: {}".format(opts.dataset))
