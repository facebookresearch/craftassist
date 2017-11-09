import csv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.misc import imread


def parse_csv(csv_fname):
    label_list = {}
    userid_label = {}
    with open(csv_fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for idx, row in enumerate(csv_reader):
            # worker-id
            if idx > 0:
                userid = row[15]
                if userid not in userid_label:
                    userid_label[userid] = []
                # load and show image
                fn = row[27]
                userid_label[userid].append(fn)
                fn = fn.split("/")
                try:
                    fn = os.path.join(im_folder, fn[-2], fn[-1])
                    if fn not in label_list:
                        label_list[fn] = []
                except IndexError:
                    continue
                # parse polygons
                try:
                    polygons = eval(row[30])
                    label_list[fn].append(polygons)
                except NameError:
                    pass
    for userid in userid_label.keys():
        print(userid, len(userid_label[userid]))
    return label_list


def color_map(label):
    if label.startswith("Mountain"):
        return "g"  # green
    elif label.startswith("Hill"):
        return "b"  # blue
    elif label.startswith("Beach"):
        return "y"  # Yellow
    elif label.startswith("Cliff"):
        return "r"  # Red
    elif label.startswith("Forest"):
        return "m"  # Magenta
    elif label.startswith("other"):
        return "w"
    else:
        raise TypeError("unknown label")


def visualize(fn, polygons_list):
    if os.path.exists(fn):
        im = imread(fn)
        # fig, ax = plt.subplots(2, 2, 1)
        ax = plt.subplot(1, 2, 1)

        ax.imshow(im)
        # ax.title(fn)

        # polygon
        for id_, polygons in enumerate(polygons_list, start=2):  # range(2, 5):
            ax = plt.subplot(1, 2, id_)
            ax.imshow(im)
            for polygon in polygons:
                vertices = polygon["vertices"]
                poly = np.ndarray((len(vertices), 2))
                label = polygon["label"]
                c = color_map(label)
                for i, vertice in enumerate(vertices):
                    poly[i][0] = vertice["x"]
                    poly[i][1] = vertice["y"]
                mpoly = patches.Polygon(poly, color=c, alpha=0.5)
                ax.add_patch(mpoly)

        plt.show()


if __name__ == "__main__":
    im_folder = "../../gather_latest/"
    # csv_fn = "Batch_3591177_batch_results.csv"
    # csv_fn = "Batch_233812_batch_results.csv"
    csv_fn = "Batch_3619652_batch_results.csv"
    fn_list = parse_csv(csv_fn)
    keys = list(fn_list.keys())
    random.shuffle(keys)
    for k in keys:
        for item in fn_list[k]:
            visualize(k, [item])
