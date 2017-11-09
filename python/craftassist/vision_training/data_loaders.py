import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import multiprocessing
import mc_block_ids as mbi
from center_voxel_cube import get_voxel_center_cube
from abc import ABC, abstractmethod


class CenterVoxelCubeLoader(ABC):
    def __init__(self, training_dirs):
        assert isinstance(training_dirs, list)
        self.training_dirs = training_dirs

    @abstractmethod
    def _get_target(self, k):
        """
        This function implements how to generate the target
        vector for the loss function. Different losses have
        different requirements for the target format.
        """
        pass

    @abstractmethod
    def loss_fn(self, gpu_id, x, y):
        """
        Return the loss function.
        """
        pass

    @abstractmethod
    def pred_fn(self, yhat):
        """
        Given a network output, this function predicts class(es).
        """
        pass

    def _get_tag_weights(self, gpu_id):
        if gpu_id >= 0:
            if self.cuda_tag_weights is None:
                self.cuda_tag_weights = self.tag_weights.cuda(gpu_id)
            return self.cuda_tag_weights
        else:
            return self.tag_weights

    def _negative_sampling_cubes(self, cube_size, train_val_ratio, nothing_n):
        """
        Sample 'nothing_n' cubes of 'cube_size' with a training/validation ratio
        of train_val_ratio.
        """

        def sample_noisy_cube(air_ratio):
            """A part should follow some nice unknown distribution. A pure random
               cube should be 'nothing'.
            """
            ## only sample up to 4 different voxels in a cube
            candidates = np.random.choice(
                range(mbi.total_ids_n()), size=random.choice([1, 2, 3, 4])
            )
            cube = np.random.choice(candidates, size=(cube_size, cube_size, cube_size, 4)).astype(
                np.float32
            )
            air_mask = np.random.choice([0, 1], size=cube.shape, p=[air_ratio, 1 - air_ratio])
            cube *= air_mask
            cube[cube_size // 2, cube_size // 2, cube_size // 2, 0] = np.random.choice(candidates)
            ## assign random (x,y,z) to the cube
            size_scale = 1.0 / 100
            cube[:, :, :, 1:] = np.random.rand(3) * size_scale
            return cube

        cubes = []
        for air_ratio in np.arange(0, 1, 0.1):
            cubes += [sample_noisy_cube(air_ratio) for i in range(int(nothing_n * 0.1))]

        random.shuffle(cubes)
        vd = cubes[: int(len(cubes) * 1.0 / (train_val_ratio + 1))]
        td = cubes[len(vd) :]
        return td, vd

    def _cube_f(self, anno):
        schematic, t, loc, cube_size = anno
        return (t, get_voxel_center_cube(loc, schematic, cube_size))

    def load_data(self, top_k, cube_size):
        def merge_tag_cubes(x, pkl):
            with open(pkl, "rb") as f:
                y = pickle.load(f)

            annos = [(schematic, t, loc, cube_size) for schematic, anno in y for t, loc in anno]
            pool = multiprocessing.Pool(20)
            cubes = pool.map(self._cube_f, annos)
            pool.close()
            pool.join()
            for t, cube in cubes:
                if t in x:
                    x[t].append(cube)
                else:
                    x[t] = [cube]

        def merge_minor_labels(minor_labels, cubes):
            minor_cubes = [c for ml in minor_labels for c in cubes.get(ml, [])]
            for ml in minor_labels:
                cubes.pop(ml, None)
            cubes["nothing"] = minor_cubes

        training_cubes, validation_cubes = {}, {}
        for d in self.training_dirs:
            merge_tag_cubes(training_cubes, d + "/training_data.pkl")
            merge_tag_cubes(validation_cubes, d + "/validation_data.pkl")

        self.tags = training_cubes.keys()
        self.tags = sorted(self.tags, key=lambda t: -len(training_cubes[t]))

        if top_k is not None:
            minor_tags = self.tags[top_k:]
            merge_minor_labels(minor_tags, training_cubes)
            merge_minor_labels(minor_tags, validation_cubes)
            self.tags = self.tags[:top_k] + ["nothing"]

        tags_n = [len(training_cubes[t]) for t in self.tags]
        logging.info("Tags: {}".format(list(zip(self.tags, tags_n))))

        ## load training and validation data
        training_data = []
        validation_data = []
        for k, t in enumerate(self.tags):
            td = training_cubes[t]
            vd = validation_cubes[t]
            for i, d in enumerate(td + vd):
                if i < len(td):
                    data = training_data
                else:
                    data = validation_data
                data.append((d, self._get_target(k)))

        ## because the training data might be imbalanced, we need to assign
        ## weights
        max_tags_n = max(tags_n)
        tag_weights = np.array([max_tags_n / float(n) for n in tags_n])
        logging.info(tags_n)
        logging.info(tag_weights)

        self.tag_weights = torch.from_numpy(tag_weights)
        self.cuda_tag_weights = None

        return training_data, validation_data


class SoftmaxLoader(CenterVoxelCubeLoader):
    def __init__(self, training_dirs=[], pred_topk=1):
        super(SoftmaxLoader, self).__init__(training_dirs)
        self.pred_topk = pred_topk

    def _get_target(self, k):
        return np.array([k]).astype(np.int64)

    def loss_fn(self, gpu_id, x, y):
        weights = self._get_tag_weights(gpu_id)
        return nn.CrossEntropyLoss(weight=weights)(x, y.view(-1))

    def pred_fn(self, yhat):
        _, idx = torch.topk(yhat, self.pred_topk, dim=-1)
        return idx


class BCELoader(CenterVoxelCubeLoader):
    def __init__(self, training_dirs=[], pred_threshold=0.8):
        super(BCELoader, self).__init__(training_dir)
        self.pred_threshold = pred_threshold

    def _get_target(self, k):
        vec = np.zeros(len(self.tags), dtype=np.float64)
        vec[k] = 1
        return vec

    def loss_fn(self, gpu_id, x, y):
        weights = self._get_tag_weights(gpu_id)
        return nn.MultiLabelSoftMarginLoss(weight=weights)(x, y)

    def pred_fn(self, yhat):
        return torch.sigmoid(yhat) > self.pred_threshold


class MultiLabelMarginLoader(CenterVoxelCubeLoader):
    def __init__(self, training_dirs=[]):
        super(MultiLabelMarginLoader, self).__init__(training_dirs)

    def _get_target(self, k):
        ## only the first non-negative indices will be treated as
        ## positive classes
        vec = np.ones(len(self.tags), dtype=np.int64) * -1
        vec[0] = k
        return vec

    def loss_fn(self, gpu_id, x, y):
        assert False, "Unsupported currently!"

    def pred_fn(self, yhat):
        assert False, "Unsupported currently!"


class CRFDataLoader(object):
    def __init__(self, training_dirs):
        assert isinstance(training_dirs, list)
        self.training_dirs = training_dirs

    def load_data(self, voxel_model, batch_size):
        """
        Output:
        batch_voxels - a list of (pred_unary, embeddings) pairs
        batch_gt_labels - a list of gt_labels

        They will be directly input to train() of a crf model
        """
        data = []
        for d in self.training_dirs:
            with open(d + "/validation_data.pkl", "rb") as f:
                data += pickle.load(f)

        def load_single_house(schematic, anno):
            unary, features = prepare_unary_and_features(schematic, voxel_model, batch_size)
            ## for gt_labels, we will ignore those rare ones that are not included in
            ## voxel_model.tags
            gt_labels = {
                tuple(loc): voxel_model.tags.index(tag)
                for tag, loc in anno
                if tag in voxel_model.tags
            }
            return (unary, features), gt_labels

        return zip(*map(load_single_house, *zip(*data)))
