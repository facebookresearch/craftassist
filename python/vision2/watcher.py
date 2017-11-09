#!/usr/bin/python

import logging
from multiprocessing import Queue, Process
import sys
import mc_block_ids as mbi

sys.path.append("..")
import build_utils as bu
from voxel_cnn_base import load_voxel_cnn


class SubComponentClassifier(Process):
    """
    A classifier class that calls a voxel model to output object tags.
    """

    def __init__(
        self, voxel_model_path=None, voxel_model_class=None, voxel_model_opts={}, batch_size=128
    ):
        super().__init__()

        if voxel_model_path is not None:
            logging.info(
                "SubComponentClassifier using voxel_model_path={}".format(voxel_model_path)
            )
            self.model = load_voxel_cnn(voxel_model_path, voxel_model_opts)
        else:
            assert voxel_model_class is not None
            self.voxel_model_class = voxel_model_class
            self.voxel_model_opts = voxel_model_opts

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
        Input: a list of tuples, where each tuple is ((x, y, z), [bid, mid]). This list
               represents a block object.
        Output: a dict of (loc, [tag1, tag2, ..]) pairs for all non-air blocks.
        """

        def get_tags(p):
            """
            convert a list of tag indices to a list of tags
            """
            return [self.model.tags[i][0] for i in p]

        def apply_offsets(cube_loc, offsets):
            """
            Convert the cube location back to world location
            """
            return (cube_loc[0] + offsets[0], cube_loc[1] + offsets[1], cube_loc[2] + offsets[2])

        np_blocks, offsets = bu.blocks_list_to_npy(
            agent=None, blocks=tuple_blocks, embed=False, xyz=True
        )

        pred = self.model.segment_object(np_blocks, self.batch_size)

        # convert prediction results to string tags
        return dict([(apply_offsets(loc, offsets), get_tags([p])) for loc, p in pred.items()])

    def recognize(self, list_of_tuple_blocks):
        """
        Multiple calls to _watch_single_object
        """
        tags = dict()
        for tb in list_of_tuple_blocks:
            tags.update(self._watch_single_object(tb))
        return tags

    def train_voxel(self, data_loader, save_model_path, topk, lr=1e-4, max_epochs=100):
        """
        Load data and train the vision model.
        """
        logging.info("Total voxel ids: {}".format(mbi.total_ids_n()))

        training_data, validation_data, tags = data_loader.load_data(topk)
        self.model = self.voxel_model_class(tags=tags, opts=self.voxel_model_opts)
        self.model.train_epochs(
            training_data, validation_data, save_model_path, lr, self.batch_size, max_epochs
        )
