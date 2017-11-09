import torch
import torch.utils.data
import numpy as np

# import datetime
# from tqdm import tqdm
# from IPython import embed

""" Dataset Class - Loader """


class RelposDataset(torch.utils.data.Dataset):
    def __init__(self, config, raw_data, is_train):
        self.house_ids = []
        self.component_ids = []
        for i, house in enumerate(raw_data):
            num_component = len(house)
            if num_component <= 1:
                print("[warning] house has <=1 component.")
                continue
            self.house_ids.extend([i] * (num_component - 1))
            self.component_ids.extend(list(range(1, num_component)))
        self.data_size = len(self.house_ids)
        assert len(self.house_ids) == len(self.component_ids)
        print("data_size=", self.data_size)
        print("raw_data.size=", len(raw_data))
        self.house_ids = np.array(self.house_ids)
        self.component_ids = np.array(self.component_ids)

    def __getitem__(self, index):
        return (self.house_ids[index], self.component_ids[index])

    def __len__(self):
        return self.data_size


""" Data Class - Root """


class RelposData:
    def __init__(self, config, data_file, is_train):
        self.config = config
        self.data_file = data_file
        self.is_train = is_train
        self.k_neg = self.config.args.k_neg
        self.use_schematic = self.config.args.use_schematic
        self.boxsize = self.config.args.boxsize

        self.raw_data = np.load(data_file)
        self.dataset = RelposDataset(self.config, self.raw_data, self.is_train)
        if is_train:
            self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.args.batch_size,
                shuffle=True,
                **self.config.kwargs
            )
        else:
            # TODO(demiguo): we can use a larger batch size
            self.loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.config.args.batch_size, **config.kwargs
            )

    """ Given an index example, return processed visualization schematic example (final house) """

    def process_final_example(self, house_data, cid):
        # cid component is always orange; other components are (35, 2-15)
        def get_meta(ind):
            return ind % 14 + 2 if ind != cid else 1

        seg_data = []
        inf = 1000000
        xmin, ymin, zmin = inf, inf, inf
        xmax, ymax, zmax = -inf, -inf, -inf

        for i in range(len(house_data)):
            xyzs = house_data[i][4][1]
            for (x, y, z) in xyzs:
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                zmin = min(zmin, z)
                xmax = max(xmax, x)
                ymax = max(ymax, y)
                zmax = max(zmax, z)
                seg_data.append((x, y, z, 35, get_meta(i)))

        dims = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1, 2]
        mat = np.zeros(dims, dtype="uint8")
        for block in seg_data:
            x, y, z, bid, meta = block
            mat[x - xmin, y - ymin, z - zmin, :] = bid, meta
        npy_schematic = np.transpose(mat, (1, 2, 0, 3))
        return npy_schematic

    """ Given an index example, return processed visualization schematic example (local view) """

    def process_local_example(self, house_data, cid, tvec):
        # this is what the network will see (boxsize x boxsize x boxsize)
        seg_data = []
        inf = 100000
        xmin, ymin, zmin = inf, inf, inf
        xmax, ymax, zmax = -inf, -inf, -inf
        tx, ty, tz = tvec
        cx, cy, cz = house_data[cid][1]  # old centroid
        ox, oy, oz = tx + cx, ty + cy, tz + cz  # new centroid  (center of the box)
        half_size = self.boxsize // 2  # origin /pm half_size

        # NB(demiguo): preoccupied (35, 1); current component (35, 2)
        # previous components
        for i in range(cid):
            xyzs = house_data[i][4][1]
            for (x, y, z) in xyzs:
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                zmin = min(zmin, z)
                xmax = max(xmax, x)
                ymax = max(ymax, y)
                zmax = max(zmax, z)
                seg_data.append((x, y, z, 35, 1))
        # current component
        xyzs = house_data[cid][4][1]
        for (x, y, z) in xyzs:
            x += tx
            y += ty
            z += tz
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
            seg_data.append((x, y, z, 35, 2))

        dims = [self.boxsize, self.boxsize, self.boxsize, 2]
        mat = np.zeros(dims, dtype="uint8")

        def transform_xyz(x, y, z):
            return x - ox + half_size, y - oy + half_size, z - oz + half_size

        def is_inbox(x, y, z):
            return (
                x >= 0
                and y >= 0
                and z >= 0
                and x <= 2 * half_size
                and y <= 2 * half_size
                and z <= 2 * half_size
            )

        for block in seg_data:
            x, y, z, bid, meta = block
            x, y, z = transform_xyz(x, y, z)
            if is_inbox(x, y, z):
                mat[x, y, z, :] = bid, meta
        npy_schematic = np.transpose(mat, (1, 2, 0, 3))
        return npy_schematic

    """ Given an index example, return processed visualization schematic example (global view) """

    def process_global_example(self, house_data, cid, tvec):
        seg_data = []
        inf = 100000
        xmin, ymin, zmin = inf, inf, inf
        xmax, ymax, zmax = -inf, -inf, -inf
        tx, ty, tz = tvec

        # NB(demiguo): preoccupied (35, 1); current component (35, 2)
        # previous components
        for i in range(cid):
            xyzs = house_data[i][4][1]
            for (x, y, z) in xyzs:
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                zmin = min(zmin, z)
                xmax = max(xmax, x)
                ymax = max(ymax, y)
                zmax = max(zmax, z)
                seg_data.append((x, y, z, 35, 1))
        # current component
        xyzs = house_data[cid][4][1]
        for (x, y, z) in xyzs:
            x += tx
            y += ty
            z += tz
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
            seg_data.append((x, y, z, 35, 2))

        dims = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1, 2]
        mat = np.zeros(dims, dtype="uint8")
        for block in seg_data:
            x, y, z, bid, meta = block
            mat[x - xmin, y - ymin, z - zmin, :] = bid, meta
        npy_schematic = np.transpose(mat, (1, 2, 0, 3))
        return npy_schematic

    """ Given an index example, return processed data example """

    def process_data_example(self, house_data, cid, tvec):
        # house data: list of component data
        # (label_id, centroid, positive_examples, negative_examples,
        #    component(label,xyzs), raw_file)
        # cid: component id
        # tvec: translation vector
        # return a LongTensor [boxsize, boxsize, boxsize]

        # sanity checks
        assert not self.use_schematic, "now only support use_schematic=False"
        assert len(house_data) > cid
        assert self.boxsize % 2 == 1, "now only support odd box size"
        assert (
            self.use_schematic or self.config.vsize == 3
        ), "if not use_schematic, vocab size equals to 3"

        tx, ty, tz = tvec
        cx, cy, cz = house_data[cid][1]  # old centroid
        ox, oy, oz = tx + cx, ty + cy, tz + cz  # new centroid  (center of the box)
        half_size = self.boxsize // 2  # origin /pm half_size

        # use_schematic = False
        # 0: empty; 1: occupied; 2: current component
        box = np.zeros((self.boxsize, self.boxsize, self.boxsize), dtype=np.int32)

        def transform_xyz(x, y, z):
            return x - ox + half_size, y - oy + half_size, z - oz + half_size

        def is_inbox(x, y, z):
            return (
                x >= 0
                and y >= 0
                and z >= 0
                and x <= 2 * half_size
                and y <= 2 * half_size
                and z <= 2 * half_size
            )

        # fill in components before the current component
        for i in range(cid):
            xyzs = house_data[i][4][1]
            for (x, y, z) in xyzs:
                nx, ny, nz = transform_xyz(x, y, z)
                if is_inbox(nx, ny, nz):
                    box[nx][ny][nz] = 1

        # fill in current component
        xyzs = house_data[cid][4][1]
        for (x, y, z) in xyzs:
            nx, ny, nz = transform_xyz(tx + x, ty + y, tz + z)
            if is_inbox(nx, ny, nz):
                box[nx][ny][nz] = 2
        return box

    """ Given an index batch, return a positive and negative data batch """

    def process_data_batch(self, batch_data, visualize=False):
        # positive data batch: LongTensor [batch_size, boxsize, boxsize, boxsize]
        #                      If not use_schematic,
        #                      then use 0/1/2 to represent empty/occupied/current
        # negative data batch: LongTensor [batch_size, k_neg, boxsize, boxsize, boxsize]
        # NB(demiguo): cpu tensors
        batch_house_ids, batch_component_ids = batch_data
        self.batch_size = batch_house_ids.size(0)
        positive_data = []
        negative_data = []
        if visualize:
            positive_global = []
            negative_global = []
            final_schematic = []
            positive_local = []
            negative_local = []

        # TODO(demiguo): check if we have enough negative examples
        min_neg = self.config.args.k_neg
        for i in range(batch_house_ids.size(0)):
            hid = batch_house_ids[i]
            cid = batch_component_ids[i]
            min_neg = min(min_neg, len(self.raw_data[hid][cid][3]))
        if min_neg < self.config.args.k_neg:
            self.config.log.warning("min_neg:%d < k_neg:%d" % (min_neg, self.config.args.k_neg))
        assert min_neg > 0, "not enough negative examples"

        for i in range(batch_house_ids.size(0)):
            hid = batch_house_ids[i]
            cid = batch_component_ids[i]
            # TODO(demiguo): get examples on-the-fly -> this might be slow

            # get house schematic
            if visualize:
                final_schematic.append(self.process_final_example(self.raw_data[hid], cid))

            # get positive example
            positive_data.append(self.process_data_example(self.raw_data[hid], cid, (0, 0, 0)))
            if visualize:
                positive_global.append(
                    self.process_global_example(self.raw_data[hid], cid, (0, 0, 0))
                )
                positive_local.append(
                    self.process_local_example(self.raw_data[hid], cid, (0, 0, 0))
                )

            # get negative examples
            # NB(demiguo): always use first min_neg examples
            if visualize:
                negative_global_per_example = []
                negative_local_per_example = []
            for j in range(min_neg):
                x, y, z = self.raw_data[hid][cid][3][j]
                negative_data.append(self.process_data_example(self.raw_data[hid], cid, (x, y, z)))
                assert x != 0 or y != 0 or z != 0, "[fatal] invalid negative example"
                if visualize:
                    negative_global_per_example.append(
                        self.process_global_example(self.raw_data[hid], cid, (x, y, z))
                    )
                    negative_local_per_example.append(
                        self.process_local_example(self.raw_data[hid], cid, (x, y, z))
                    )
            if visualize:
                negative_global.append(negative_global_per_example)
                negative_local.append(negative_local_per_example)
        positive_data = torch.LongTensor(positive_data)
        negative_data = torch.LongTensor(negative_data)
        assert positive_data.size() == (self.batch_size, self.boxsize, self.boxsize, self.boxsize)
        negative_data = negative_data.contiguous().view(
            self.batch_size, min_neg, self.boxsize, self.boxsize, self.boxsize
        )
        if visualize:
            return (
                positive_data,
                negative_data,
                positive_global,
                negative_global,
                positive_local,
                negative_local,
                final_schematic,
            )
        else:
            return positive_data, negative_data
