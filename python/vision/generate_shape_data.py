import numpy as np
import os
import sys

VISION_DIR = os.path.dirname(os.path.realpath(__file__))
PYTHON_DIR = os.path.join(VISION_DIR, "../")
STACK_AGENT_DIR = os.path.join(PYTHON_DIR, "craftassist")
sys.path.append(STACK_AGENT_DIR)
import shapes

SHAPENAMES = ["cube", "rectanguloid", "sphere", "pyramid", "tower", "empty"]


def empty(args):
    num = np.random.randint(1, 6)
    S = []
    for i in range(num):
        pos = np.random.randint(0, 10, 3)
        bid = np.random.randint(0, 64)
        S.append((pos, bid))
    return S


SHAPEFNS = [
    shapes.cube,
    shapes.rectanguloid,
    shapes.sphere,
    shapes.square_pyramid,
    shapes.tower,
    empty,
]


def transpose_convert_to_np(blocks):
    bounds = shapes.get_bounds(blocks)
    dX = bounds[1] - bounds[0]
    dY = bounds[3] - bounds[2]
    dZ = bounds[5] - bounds[4]
    #    if dX<=0 or dY<=0 or dZ<=0:
    #        import pdb
    #        pdb.set_trace()
    out = np.zeros((dY + 1, dZ + 1, dX + 1, 2), dtype="int32")
    for b in blocks:
        x = b[0][0] - bounds[0]
        y = b[0][1] - bounds[2]
        z = b[0][2] - bounds[4]
        if type(b[1]) is int:
            out[y, z, x, 0] = b[1]
        else:
            out[y, z, x, 0] = b[1][0]
            out[y, z, x, 1] = b[1][1]
    return out


def get_random_args():
    args = {}
    i = np.random.randint(0, 6)
    if i == 0:
        args["size"] = np.random.randint(5, 20)
    elif i == 1:
        args["sizes"] = np.random.randint(5, 20, size=3)
    elif i == 2:
        args["radius"] = np.random.randint(5, 20)
    elif i == 3:
        args["base_radius"] = np.random.randint(5, 20)
        args["slope"] = np.random.rand() * 0.4 + 0.8
        fullheight = args["base_radius"] * args["slope"]
        args["height"] = np.random.randint(0.5 * fullheight, fullheight)
    elif i == 4:
        args["height"] = np.random.randint(3, 20)
        args["base"] = np.random.randint(0, 2)

    args["bid"] = np.random.randint(1, 64)
    return SHAPEFNS[i], args, SHAPENAMES[i]


if __name__ == "__main__":
    import os
    import argparse
    from tqdm import tqdm

    sys.path.append(PYTHON_DIR)
    import render_schematic

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", default="/private/home/aszlam/junk/shapes/")
    parser.add_argument("--shapes", type=int, default=20000)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument(
        "--render_target_path", default="/private/home/aszlam/junk/rendered_shapes/"
    )
    opts = parser.parse_args()

    if opts.render == 1:
        import shutil

        temp_dir = "/scratch/render_junk/"
        yaws = [0, 20, 45, 65]
    #        angles = ['0','20', '45', '65']
    # all of these are symmetric, so views between 0 and 90 are enough. FIXME

    for n in SHAPENAMES:
        os.makedirs(opts.target_path + n + "/", exist_ok=True)
        if opts.render == 1:
            os.makedirs(opts.render_target_path + n + "/", exist_ok=True)

    for i in tqdm(range(opts.shapes)):
        fn, args, name = get_random_args()
        blocks = fn(args)
        yzxm = transpose_convert_to_np(blocks)
        spath = opts.target_path + name + "/" + str(i) + ".npy"
        np.save(spath, yzxm)
        if opts.render == 1:
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir)
            render_schematic.render(
                spath, temp_dir, True, 25564, distance=np.random.randint(3, 64), yaws=yaws
            )
            for a in yaws:
                src = "block." + str(a) + ".bin"
                dst = str(i) + src
                src = temp_dir + src
                dst = opts.render_target_path + name + "/" + dst
                print(dst)
                try:
                    shutil.copyfile(src, dst)
                except:
                    pass
