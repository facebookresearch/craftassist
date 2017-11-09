import json
import random
import numpy as np
from tqdm import tqdm

from .utils import data2dic, process_break
from .utils import in_neighbor
from .utils import get_data, get_targets


def read_houses(file_list, data_order="human"):
    # file_list: a list of str containing file names
    # return processed houses and file-list
    houses, fn_list = [], []

    for fn in file_list:
        # data is a list of atomic actions
        # [tick-time, entity, [x, y, z], [block-id, meta], 'P/B']
        house_raw = json.load(open(fn, "r"))
        _, house_clean = process_break(house_raw, fn, data_order=data_order)
        if len(house_clean) >= 100:
            houses.append(house_clean)
            fn_list.append(fn)
    return houses, fn_list


class HouseDatasetOrder(object):
    """
    Data loader of houses to generate in human order
      file_list: a list of str, containing all .json filenames
      batch_size:
      block_id: category number of materials, 1 for binary classification; 256 for normal houses
      local_block_size: of shape (block_id, block_size, block_size, block_size)
      global_block_size: of shape (block_id, block_size, block_size, block_size)
     """

    # Load all the houses (json files) available from the folder
    def __init__(self, file_list, args, data_order="human"):
        print("loading houses...")
        self.houses, self.fn_list = read_houses(file_list, data_order)
        self.house_num = len(self.houses)
        self.step_num = sum([len(item) for item in self.houses])

        self.block_id = args.block_id  # block_id
        self.global_block_id = args.global_block_id
        self.local_bsize = args.local_bsize
        self.global_bsize = args.global_bsize
        self.batch_size = args.batch_size
        self.history = args.history
        self.embed = args.embed
        self.multi_res = args.multi_res
        self.train_target_order = args.train_target_order
        self.total_steps = 0
        self.useful_steps = 0

        self.id_data = 0  # outer-loop: image-id
        self.id_step = 0  # inner-loop: step-id
        self.finish = False  # have we gone through all the data?

        total_steps = np.sum([len(house) for house in self.houses])
        print("total_steps=", total_steps)

    def reset(self):
        self.id_data = 0
        self.id_step = 0
        self.finish = False

    def get_frequencies(self, filename="saved/block_type_freq"):
        local_bsize = self.local_bsize
        h_size = int((local_bsize - 1) / 2)  # half window
        uncond_counter = np.zeros((256))
        joint_counter = np.zeros((256, 256))  # (a, b) : a first b second
        for i in tqdm(range(len(self.houses)), desc="get freq"):
            house = self.houses[i]
            for t in range(len(house) - 1):
                xyz_t = house[t][2]
                xyz_tp1 = house[t + 1][2]

                if not in_neighbor(xyz_t, xyz_tp1, h_size):
                    continue  # this is ignored in both train and validation

                b_t = house[t][3][0]
                b_tp1 = house[t + 1][3][0]
                assert b_t >= 0 and b_tp1 >= 0
                uncond_counter[b_tp1] += 1
                joint_counter[b_t][b_tp1] += 1
        f = open(filename, "w")
        uncond_tot = np.sum(uncond_counter)
        for b in range(256):
            if uncond_counter[b] > 0:
                f.write("p(%d)=%.3lf\n" % (b, uncond_counter[b] * 1.0 / uncond_tot))
        f.write("\n\n")
        for b1 in range(256):
            cond_tot = np.sum(joint_counter[b1])
            for b2 in range(256):
                if joint_counter[b1][b2] != 0:
                    f.write("p(%d|%d)=%.3lf\n" % (b2, b1, 1.0 * joint_counter[b1][b2] / cond_tot))
            f.write("\n")

        # how many x->x transitions?
        tot_same = 0
        for b in range(256):
            tot_same += joint_counter[b][b]
        f.write(
            "p(x|x): %d, percentage %.3lf\n" % (tot_same, tot_same * 1.0 / np.sum(joint_counter))
        )
        f.close()

    def sample(self, train=True):
        local_bsize = self.local_bsize
        h_size = int((local_bsize - 1) / 2)  # half window
        global_bsize = self.global_bsize
        house = self.houses[self.id_data]  # current house

        # sample a step t, where next action is within neighborhood
        while True:
            self.total_steps += 1
            if train:
                t = random.randint(0, len(house) - 2)
            else:
                t = self.id_step
                if t == len(house) - 1:
                    self.id_data += 1
                    self.id_step = 0
                    if self.id_data == len(self.houses):
                        self.finish = True
                        return None, None, (None, None, None, None)
                    else:
                        house = self.houses[self.id_data]
                        continue
                self.id_step += 1

            xyz_t = house[t][2]
            found_nt = -1

            def raster_cmp(xyz1, xyz2):
                # NB(demi): y is height in minecraft
                yxz1 = (xyz1[1], xyz1[0], xyz1[2])
                yxz2 = (xyz2[1], xyz2[0], xyz2[2])
                return yxz1 < yxz2

            if self.train_target_order == "raster_scan_global":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                        found_nt = nt
                if found_nt != -1 and in_neighbor(xyz_t, house[found_nt][2], h_size):
                    break
            elif self.train_target_order == "raster_scan_global_suffix":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if raster_cmp(xyz_t, house[nt][2]):
                        if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                            found_nt = nt
                if found_nt != -1 and in_neighbor(xyz_t, house[found_nt][2], h_size):
                    break

            elif self.train_target_order == "raster_scan_local":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if in_neighbor(xyz_t, house[nt][2], h_size):
                        if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                            found_nt = nt
                if found_nt != -1:
                    break
            elif self.train_target_order == "raster_scan_local_suffix":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if in_neighbor(xyz_t, house[nt][2], h_size) and raster_cmp(
                        xyz_t, house[nt][2]
                    ):
                        if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                            found_nt = nt
                if found_nt != -1:
                    break
            elif self.train_target_order == "raster_scan_global_suffix_inter":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if raster_cmp(xyz_t, house[nt][2]):
                        if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                            found_nt = nt
                if found_nt != -1 and in_neighbor(xyz_t, house[found_nt][2], h_size):
                    if in_neighbor(xyz_t, house[t + 1][2], h_size):
                        break

            elif self.train_target_order == "raster_scan_local_suffix_inter":
                found_nt = -1
                for nt in range(t + 1, len(house)):
                    if in_neighbor(xyz_t, house[nt][2], h_size) and raster_cmp(
                        xyz_t, house[nt][2]
                    ):
                        if found_nt == -1 or raster_cmp(house[nt][2], house[found_nt][2]):
                            found_nt = nt
                if found_nt != -1 and in_neighbor(xyz_t, house[t + 1][2], h_size):
                    break
            else:
                assert self.train_target_order == "default"
                xyz_tp1 = house[t + 1][2]
                if in_neighbor(xyz_t, xyz_tp1, h_size):
                    found_nt = t + 1
                    break
        self.useful_steps += 1
        if train:
            self.id_data = (self.id_data + 1) % self.house_num
        local_data = get_data(house, t, local_bsize, self.block_id, self.history, embed=self.embed)
        if self.multi_res:
            global_data = get_data(
                house, t, global_bsize, self.global_block_id, 1, embed=self.embed
            )
        else:
            global_data = None
        return (
            local_data,
            global_data,
            get_targets(
                house,
                t,
                found_nt,
                local_bsize,
                self.block_id,
                predict_next_steps=True,
                no_groundtruth=False,
            ),
        )

    def sample_gen(self, id_, finished_perct=None):
        # select house
        house = self.houses[id_]

        # select a step
        if finished_perct is None:
            t = random.randint(0, len(house) - 2)
        else:
            t = int(len(house) * finished_perct)
            assert t < len(house)
        half_house = house[: t + 1]

        # dictionary: to-do
        coords_done = data2dic(house[: t + 1])
        coords_todo = data2dic(house[t + 1 :])

        return half_house, t, coords_todo, coords_done

    def iterate(self, train=True):
        local_bsize = self.local_bsize
        global_bsize = self.global_bsize
        local_data = []
        global_data = []
        target1 = []
        target2 = []
        target_next_steps_coord = []
        target_next_steps_type = []
        # TODO(demiguo): return local and global house data
        for i in range(self.batch_size):
            # get one example from self.current_id
            ldata, gdata, (t1, t2, t_coord, t_type) = self.sample(train)
            local_data.append(ldata)
            global_data.append(gdata)
            target1.append(t1)
            target2.append(t2)
            # gdata will also be None if multi-res is disabled
            # so only check ldata now
            if ldata is None:
                return None, None, None, None, None
            target_next_steps_coord.append(t_coord)
            target_next_steps_type.append(t_type)
        if not self.embed:
            local_data = np.vstack(local_data).reshape(
                self.batch_size,
                self.block_id * self.history,
                local_bsize,
                local_bsize,
                local_bsize,
            )
            if self.multi_res:
                global_data = np.vstack(global_data).reshape(
                    self.batch_size, self.global_block_id, global_bsize, global_bsize, global_bsize
                )
            else:
                global_data = []
        else:
            local_data = np.vstack(local_data).reshape(
                self.batch_size, self.history, local_bsize, local_bsize, local_bsize
            )
            if self.multi_res:
                global_data = np.vstack(global_data).reshape(
                    self.batch_size, 1, global_bsize, global_bsize, global_bsize
                )

        target1 = np.array(target1)
        target2 = np.array(target2)
        target_next_steps = (np.array(target_next_steps_coord), np.array(target_next_steps_type))
        return local_data, global_data, target1, target2, target_next_steps
