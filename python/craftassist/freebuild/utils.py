import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import shutil


def in_neighbor(xyz, xyz2, threshold):
    for x, x2 in zip(xyz, xyz2):
        if abs(x - x2) > threshold:
            return False
    return True


def sync_file():
    """
    copy all files from fair-cluster to local houses folder
    """
    folder = "/private/home/jsgray/housebuild/out"
    files = os.listdir(folder)
    files = [item for item in files if item.startswith("workdir.")]
    files.sort()
    # copy only the placed.json file to ./data/houses
    for item in files:
        # ":" is not allowed in Windows and Mac systems
        item = item.replace(":", "-")
        dest_folder = os.path.join("./data/houses", item)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        src = os.path.join(folder, item, "placed.json")
        dst = os.path.join(dest_folder, "placed.json")
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def coord_range(data):
    # coord of first action
    x, y, z = data[0][2]

    # go through the file the first time to get x, y, z range
    x_min, x_max = x, x
    y_min, y_max = y, y
    z_min, z_max = z, z
    for item in data:
        x, y, z = item[2]
        x_min = min(x, x_min)
        x_max = max(x, x_max)
        y_min = min(y, y_min)
        y_max = max(y, y_max)
        z_min = min(z, z_min)
        z_max = max(z, z_max)
    xyz_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    return xyz_range


def data2npy(data, compress=False, end_id=-1, xyz_range=None):
    """
    data: a raw list of data from json files
      each item is [time, userid, [x,y,z], [bid,meta], 'P'/'B']
      where x,y,z: w, h, d
    compress:
      group id into 8 - 10 small categories
    end_id:
      only visualize data[:end_id]
      if end_id == -1, visualize all actions
    xyz_range:
      [x_min, y_min, z_min, x_max, y_max, z_max]
    output: a 4d numpy array of size (w, h, d, 2)
    """
    if xyz_range is None:
        print("run here")
        xyz_range = coord_range(data)
    x_min, y_min, z_min, x_max, y_max, z_max = xyz_range
    # print(x_min, y_min, z_min)
    # print(x_max, y_max, z_max)

    # dimension of (x, y, z, 2)
    dims = [x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1, 2]
    # create a 4D-numpy array
    mat = np.zeros(dims, dtype="uint8")

    for i, item in enumerate(data):
        x, y, z = item[2]
        bid, meta = item[3]
        bid = np.uint8(bid)
        x -= x_min
        y -= y_min
        z -= z_min

        # TODO: how to compress
        """ if compress:
            bid_short = full_to_simple(bid)
            bid, meta = simple_to_full(bid_short)
        """
        if data[-1] != "B":
            mat[x, y, z, :] = bid, meta
        else:
            mat[x, y, z, :] = 0, 0
        # if i == 250:
        #     break
        if end_id >= 0 and i == end_id:
            break
    return mat


def plot_3d(data, fig=None, subplot=None):
    """
    data can be either:
      a json file contains a list of
      [timestamp, user, [x, y, z], [bid, meta-id], 'P'/'B']
      atomic actions, where y is the height dimension.
    or:
      a numpy array of shape (w, h, d, ?)
    visualize with 3d-plot
    """
    import matplotlib.pyplot as plt

    if isinstance(data, list):
        coords = data2dic(data)
    elif isinstance(data, np.ndarray):
        coords = {}
        h, w, d, _ = data.shape
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    bid = data[i, j, k, 0]
                    if bid != 0:
                        coords[(i, j, k)] = 1
    else:
        raise TypeError("Unknown data type: should be numpy ndarray or a list of raw data.")

    # width, height, depth dimension
    ws, hs, ds = [], [], []
    title = "%d-points" % len(coords.keys())
    for item in coords:
        x, y, z = item
        ws.append(x)
        hs.append(y)
        ds.append(z)
    if fig is None:
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d', title=title)
        ax = Axes3D(fig)
    else:
        ax = fig.add_subplot(subplot, projection="3d", title=title)
    ax.scatter(ws, ds, hs, c="r", marker="o")
    ax.set_xlabel("W Label")
    ax.set_ylabel("D Label")
    ax.set_zlabel("H Label")
    plt.show()


def data2dic(data, all_act=False, end_id=-1):
    """
    input data loaded from json files
      if not all_act: return the last action
      if all_act: return a dict: (x,y,z) to all actions
    end_id: how many action to record
    """
    coord_2_action = {}
    for i, item in enumerate(data):
        t = item[0]
        x, y, z = item[2]
        block_id, meta = item[3]
        action = item[4]
        if all_act:
            if (x, y, z) not in coord_2_action:
                coord_2_action[(x, y, z)] = [(block_id, meta, action, t)]
            else:
                coord_2_action[(x, y, z)].append((block_id, meta, action, t))
        else:
            if action != "B":
                coord_2_action[(x, y, z)] = [block_id, meta, action, t]
            elif (x, y, z) in coord_2_action:
                del coord_2_action[(x, y, z)]
        if end_id > -1 and i == end_id:
            break
    return coord_2_action


def get_data(action_seq, step_id, bsize, num_block_type, history, embed=False):
    """ Convert sequence of actions (up to step_id) to 3D data representation. """
    h_size = int((bsize - 1) / 2)  # half window
    if embed:
        content = np.zeros([1, bsize, bsize, bsize], dtype=np.int64)
    else:
        content = np.zeros([num_block_type, bsize, bsize, bsize], dtype=np.float32)
    # center = action_seq[step_id][2] # (x, y, z) of the last step
    center = action_seq[step_id][0]  # (x, y, z) of the last step
    xc, yc, zc = center
    # encode current
    for i in range(step_id + 1):
        item = action_seq[i]
        x, y, z = item[0]
        # if abs(x-xc) <= h_size and abs(y-yc) <= h_size and abs(z-zc) <= h_size:
        if in_neighbor(center, (x, y, z), h_size):
            x_ = x - xc + h_size
            y_ = y - yc + h_size
            z_ = z - zc + h_size
            # if item[-1] == "P": # Place a block
            if num_block_type == 1:  # binarized occupancy map
                bid = 0
            else:
                # bid = (item[3][0] + 256) % 256
                bid = (item[1][0] + 256) % 256
            if embed:
                content[0, x_, y_, z_] = bid
            else:
                content[bid, x_, y_, z_] = 1.0

    # encode history
    contents = [content]
    for i in range(step_id, step_id - history + 1, -1):
        if i >= 0:
            last = np.copy(contents[-1])
            # remove step[i] if present
            # x, y, z = action_seq[i][2]
            x, y, z = action_seq[i][0]
            # if abs(x-xc) <= h_size and abs(y-yc) <= h_size and abs(z-zc) <= h_size:
            if in_neighbor((x, y, z), (xc, yc, zc), h_size):
                x_ = x - xc + h_size
                y_ = y - yc + h_size
                z_ = z - zc + h_size
                if embed:
                    last[0, x_, y_, z_] = 0
                else:
                    last[:, x_, y_, z_] = 0.0
            contents.append(last)
        else:
            contents.append(
                np.zeros(
                    [(num_block_type if not embed else 1), bsize, bsize, bsize], dtype=np.float32
                )
            )
    content = np.vstack(contents)  # [history, num_block_type, bsize, bsize, bsize]
    return content


def get_targets(
    action_seq,
    step_id,
    next_id,
    bsize,
    num_block_type,
    predict_next_steps=True,
    no_groundtruth=False,
):
    """ Convert sequence of actions (up to step_id+1) to target representations. """
    assert next_id != -1
    h_size = int((bsize - 1) / 2)  # half window

    center = action_seq[step_id][2]  # (x, y, z) of the last step
    xc, yc, zc = center

    # NB: this is not necessarily next step
    if not no_groundtruth:
        pred = action_seq[next_id]
        xp, yp, zp = pred[2]

        # organize the predicted ground truth
        xp = xp - xc + h_size
        yp = yp - yc + h_size
        zp = zp - zc + h_size
        target1 = xp * bsize ** 2 + yp * bsize + zp
        if pred[-1] == "P":
            target2 = np.uint8(pred[3][0])
        else:
            target2 = 0
    else:
        assert False
        target1 = None
        target2 = None

    # go through all following steps (always in human order)
    target_next_steps_coord = []
    target_next_steps_type = []
    if predict_next_steps:
        for i in range(step_id + 1, min(step_id + 10, len(action_seq))):
            pred = action_seq[i]
            xp, yp, zp = pred[2]
            if in_neighbor((xp, yp, zp), (xc, yc, zc), h_size):
                # if abs(xp-xc) <= h_size and abs(yp-yc) <= h_size and abs(zp-zc) <= h_size:
                xp = xp - xc + h_size
                yp = yp - yc + h_size
                zp = zp - zc + h_size
                tmp = xp * bsize ** 2 + yp * bsize + zp
                target_next_steps_coord.append(tmp)
                target_next_steps_type.append(pred[3][0])

    return target1, target2, target_next_steps_coord, target_next_steps_type


def process_break(house, fn="", data_order="human"):
    # house: a list of actions of a single house
    # return a cleaned version
    #
    #  Remove all "Break" action from the data.
    #  At the same position, if multiple actions are taken,
    #  only the last action is recorded.
    house_clean = []
    coords = data2dic(house)

    for i, item in enumerate(house[::-1]):
        t, userid, xyz, blocktype, action = item
        blocktype = ((blocktype[0] + 256) % 256, blocktype[1])
        x, y, z = xyz
        if (x, y, z) in coords and coords[(x, y, z)][2] != "B":
            _, meta, action, t = coords[(x, y, z)]
            house_clean.append([t, userid, xyz, blocktype, action])
            del coords[(x, y, z)]
    house_clean = house_clean[::-1]

    # sort by height
    # house_clean.sort(key= lambda x: (x[2][1], x[0]))
    if data_order == "human":
        pass
    elif data_order == "raster-scan":
        house_clean = sorted(house_clean, key=lambda x: (x[2][1], x[2][0], x[2][2], x[0]))
    elif data_order == "random":
        random.shuffle(house_clean)
    else:
        raise NotImplementedError
    return house, house_clean


def npy_sparse2full(npy, num_block_type=1, history=1):
    s1, s2, s3, s4 = npy.shape
    assert s4 == 2

    content = np.zeros((num_block_type, s1, s2, s3), dtype=np.float32)

    for x in range(s1):
        for y in range(s2):
            for z in range(s3):
                bid, _ = npy[x, y, z]
                if bid == 0:
                    continue

                if num_block_type == 1:  # binarized occupancy map
                    bid = 0
                else:
                    bid = npy[x, y, z, 0]

                content[bid, x, y, z] = 1.0

    contents = [content]
    for i in range(history - 1):
        last = np.copy(contents[-1])
        h_size = s1 // 2
        last[:, h_size, h_size, h_size] = 0.0
        contents.append(last)

    content = np.vstack(contents)  # [history, num_block_type, bsize, bsize, bsize]
    content = np.expand_dims(content, axis=0)
    return content
