#!/usr/bin/python

import logging
from multiprocessing import Queue, Process
import sys
import mc_block_ids as mbi

sys.path.append("..")
import build_utils as bu
from voxel_cnn_base import load_voxel_cnn
from seg_crf_models import VoxelCRF, prepare_unary_and_features


class SubComponentClassifier(Process):
    """
    A classifier class that calls a voxel model to output object tags.
    """

    def __init__(
        self, voxel_model_path=None, voxel_model_class=None, voxel_model_opts={}, batch_size=128
    ):
        super(SubComponentClassifier, self).__init__()

        if voxel_model_path is not None:
            logging.info(
                "SubComponentClassifier using voxel_model_path={}".format(voxel_model_path)
            )
            self.model = load_voxel_cnn(voxel_model_path, voxel_model_opts)
            self.crf_model = VoxelCRF(
                pairwise_weight1=30.0, sxyz1=2.0, sembed=(0.5 if mbi.voxel_group else 3.0)
            )
            assert self.model.embedding.weight.data.size()[0] == mbi.total_ids_n()
            logging.info("Support {} tags: {}".format(len(self.model.tags), self.model.tags))
        else:
            assert voxel_model_class is not None
            self.voxel_model_class = voxel_model_class
            self.voxel_model_opts = voxel_model_opts
            self.crf_model = VoxelCRF()

        ## assume that training and prediction use the same batch size
        self.batch_size = batch_size

        self.block_objs_q = Queue()  # store block objects to be recognized
        self.loc2labels_q = Queue()  # store loc2labels dicts to be retrieved by the agent
        self.daemon = True

    def run(self):
        """
        The main recognition loop of the classifier
        """
        while True:  # run forever
            tb = self.block_objs_q.get(block=True, timeout=None)
            loc2labels = self._watch_single_object(tb)
            self.loc2labels_q.put((loc2labels, tb))

    def _watch_single_object(self, tuple_blocks):
        """
        Input: a list of tuples, where each tuple is ((x, y, z), [bid, mid])
        Output: a dict of (loc, [tag1, tag2, ..]) pairs for all non-air blocks.
        """

        def get_tags(p):
            """
            convert a list of tag indices to a list of tags
            """
            return [self.model.tags[i] for i in p]

        def apply_offsets(cube_loc, offsets):
            """
            Convert the cube location back to world location
            """
            return (cube_loc[0] + offsets[0], cube_loc[1] + offsets[1], cube_loc[2] + offsets[2])

        np_blocks, offsets = bu.blocks_list_to_npy(tuple_blocks, xyz=True)
        unary, features = prepare_unary_and_features(np_blocks, self.model, self.batch_size)
        pred = self.crf_model.inference(unary, features)
        #        logging.info("CRF pred: {}".format(pred))

        # convert prediction results to string tags
        return dict([(apply_offsets(loc, offsets), get_tags([p])) for loc, p in pred.items()])

    def recognize(self, list_of_tuple_blocks):
        tags = dict()
        for tb in list_of_tuple_blocks:
            tags.update(self._watch_single_object(tb))
        return tags

    def train_voxel(self, data_loader, save_model_path, top_k=None, lr=1e-4, max_epochs=100):
        """ """
        mbi.voxel_group = self.voxel_model_opts["voxel_group"]
        logging.info("Total voxel ids: {}".format(mbi.total_ids_n()))

        training_data, validation_data = data_loader.load_data(
            top_k, self.voxel_model_class.cube_size
        )
        self.model = self.voxel_model_class(tags=data_loader.tags, opts=self.voxel_model_opts)
        self.model.train_epochs(
            training_data,
            validation_data,
            data_loader.loss_fn,
            data_loader.pred_fn,
            save_model_path,
            lr,
            self.batch_size,
            max_epochs,
        )

    def train_crf(
        self, data_loader, save_model_path, pairwise_weight_range, sxyz_range, sembed_range
    ):

        batch_voxels, batch_gt_labels = data_loader.load_data(self.model, self.batch_size)
        self.crf_model.train(
            batch_voxels,
            batch_gt_labels,
            save_model_path,
            pairwise_weight_range,
            sxyz_range,
            sembed_range,
        )
