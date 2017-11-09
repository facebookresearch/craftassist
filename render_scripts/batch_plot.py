import numpy as np
import os
import random
import shutil


def filter(depthpath):
    # given depth, decide if it is a good photo or not?
    depth = np.fromfile(depthpath, np.float32)
    edge = int((len(depth)) ** 0.5)
    depth = depth.reshape((edge, edge))
    depth[depth > 100] = 100
    center = depth[50:-50, 50:-50]
    min_ = center.min()
    if min_ < 1.0:
        return False
    if min_ > 4.0:
        return False
    return True


if __name__ == "__main__":
    cmd = "python ../python/logging_plugin/plot_vision.py "
    ids = [item for item in range(1, 12501)]
    # random.shuffle(ids)
    cnt = [0, 0]
    for i, idx in enumerate(ids):  # range(1, 101):
        for yaw in range(0, 360, 90):
            folder = "../../gather_latest/%d/" % idx
            im = os.path.join(folder, "chunky.%d.png" % yaw)
            block = os.path.join(folder, "vision.%d.block.bin" % yaw)
            depth = os.path.join(folder, "vision.%d.depth.bin" % yaw)

            call = [cmd, "--blocks", block, "--img", im, "--depth", depth]
            call = " ".join(call)
            print(call)
            os.system(call)
