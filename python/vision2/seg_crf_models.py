import pydensecrf.densecrf as dcrf
import multiprocessing
import itertools
import numpy as np
import logging
import random


class VoxelCRF(object):
    """
    Given a sparse list of voxels, each voxel with a predicted label distribution
    from a deep net, we perform segmentation with Conditional Random Field.

    The model is adapted from

    "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials",
    Philipp Krähenbühl and Vladlen Koltun, NIPS 2011

    where only the bilateral pairwise potential is used.
    """

    def __init__(
        self,
        pairwise_weight1=1.0,
        pairwise_weight2=0.0,
        sxyz1=1.0,  # low value encourages different labels for neighboring voxels
        sxyz2=1.0,
        sembed=1.0,  # low value encourages different labels for different materials
        # does not influence voxels with the same material
        n_iters=20,
        model_path=None,
    ):
        if model_path is not None:
            # load stored model options
            sds = np.load(model_path).item()
            self.pairwise_weight1 = sds["pairwise_weight1"]
            self.sxyz1 = sds["sxyz1"]
            self.pairwise_weight2 = sds["pairwise_weight2"]
            self.sxyz2 = sds["sxyz2"]
            self.sembed = sds["sembed"]
            self.n_iters = sds["n_iters"]
        else:
            self.pairwise_weight1 = pairwise_weight1
            self.sxyz1 = sxyz1  ## std for xyz
            self.pairwise_weight2 = pairwise_weight2
            self.sxyz2 = sxyz2  ## std for xyz
            self.sembed = sembed  ## std for voxel embedding
            self.n_iters = n_iters  ## number of inference iterations

    def save(self, model_path):
        sds = {}
        sds["pairwise_weight1"] = self.pairwise_weight1
        sds["sxyz1"] = self.sxyz1
        sds["pairwise_weight2"] = self.pairwise_weight2
        sds["sxyz2"] = self.sxyz2
        sds["sembed"] = self.sembed
        sds["n_iters"] = self.n_iters
        np.save(model_path, sds)

    def inference(
        self,
        pred_unary,
        embeddings,
        sxyz1=None,
        sxyz2=None,
        sembed=None,
        pairwise_weight1=None,
        pairwise_weight2=None,
    ):
        """
        Input:
        pred_unary:
            a dict of {(x,y,z) : unary} where unary is a list of length n_labels,
            representing the predicted unary potentials by the voxel CNN
        embeddings:
            a dict of {(x,y,z) : embedding} where embedding is taken from the voxel CNN

        This function performs the MAP inference given the current model parameters
            self.pairwise_weight1,2
            self.sxyz1,2
            self.sembed
        It will call pydensecrf to do the inference.

        Output: a dict of {(x,y,z) : label}
        """
        assert len(pred_unary) == len(embeddings), "voxel numbers not equal!"
        voxels = list(pred_unary.keys())
        n_nodes = len(voxels)
        n_labels = len(list(pred_unary.values())[0])

        sxyz1 = sxyz1 or self.sxyz1
        sxyz2 = sxyz2 or self.sxyz2
        sembed = sembed or self.sembed
        pairwise_weight1 = pairwise_weight1 or self.pairwise_weight1
        pairwise_weight2 = pairwise_weight2 or self.pairwise_weight2

        loc_feat = np.array(voxels).astype(np.float32)
        loc_feat = np.ascontiguousarray(loc_feat.transpose(1, 0))
        embedding_feat = np.array([embeddings[v] for v in voxels]).astype(np.float32)
        embedding_feat = np.ascontiguousarray(embedding_feat.transpose(1, 0))
        feat = np.concatenate([loc_feat / sxyz1, embedding_feat / sembed], axis=0)

        unary = np.array([pred_unary[v] for v in voxels]).astype(np.float32)
        unary = np.ascontiguousarray(unary.transpose(1, 0))

        d = dcrf.DenseCRF(n_nodes, n_labels)
        d.setUnaryEnergy(unary)
        d.addPairwiseEnergy(feat, compat=pairwise_weight1)
        if pairwise_weight2 > 0:
            d.addPairwiseEnergy(loc_feat / sxyz2, compat=pairwise_weight2)
        random.seed(0)
        q = d.inference(self.n_iters)
        map_ = np.argmax(q, axis=0)
        return {v: l for v, l in zip(voxels, map_)}

    def _evaluate(self, batch_map, batch_gt_labels, f_score_beta):
        """
        Given a batch of MAP results and the correspondng groundtruth
        labels, this function computes an average F1 score across all
        label classes.
        """
        from sklearn.metrics import fbeta_score

        pred_labels = [
            batch_map[i][v] for i, gt in enumerate(batch_gt_labels) for v, _ in gt.items()
        ]
        gt_labels = [l for gt in batch_gt_labels for v, l in gt.items()]
        return fbeta_score(gt_labels, pred_labels, average="micro", beta=f_score_beta)

    def train(
        self,
        batch_voxels,
        batch_gt_labels,
        model_path,
        pairwise_weight_range,
        sxyz_range,
        sembed_range,
        f_score_beta=1.0,
        n_procs=20,
    ):
        """
        Perform a grid search of the model parameters.

        Input:
        batch_voxels:
            a list of (pred_unary, embeddings) pairs, where pred_unary and embeddings
            have the format in self.inference()
        batch_gt_labels:
            a list of gt_labels, where gt_labels is a dict of {(x,y,z) : label}

        batch_voxels and batch_gt_labels should have the same size.
        For each gt_labels in batch_gt_labels, its voxel set should be a subset of
        the voxel set of batch_voxels, i.e., we may only evaluate a subset of voxel
        labels, when dense labels are unavailable.

        f_score_beta:
            beta in the F score formula. It controls the weight of precision.
        """
        combinations = list(itertools.product(*[sxyz_range, sembed_range, pairwise_weight_range]))
        pool = multiprocessing.Pool(n_procs)
        logging.info("Start training CRF")
        ## cannot pickle lambda; function object is needed
        trials = pool.map(
            OneTrial(batch_voxels, batch_gt_labels, f_score_beta, self.inference, self._evaluate),
            combinations,
        )
        pool.close()
        pool.join()
        best_trial = max(trials, key=lambda r: r[0])

        self.sxyz, self.sembed, self.pairwise_weight = best_trial[1]
        logging.info("Best F score: {}".format(best_trial[0]))
        logging.info(
            "best sxyz: {}, best sembed: {}, best pairwise weight: {}".format(
                self.sxyz, self.sembed, self.pairwise_weight
            )
        )

        self.save(model_path)


class OneTrial(object):
    def __init__(self, batch_voxels, batch_gt_labels, f_score_beta, infer_fn, eval_fn):
        self.batch_voxels = batch_voxels
        self.batch_gt_labels = batch_gt_labels
        self.f_score_beta = f_score_beta
        self.infer_fn = infer_fn
        self.eval_fn = eval_fn

    def __call__(self, comb):
        sxyz, sembed, pw = comb
        logging.info("Trying {} {} {}".format(sxyz, sembed, pw))
        batch_map = [
            self.infer_fn(pred_unary, embeddings, sxyz, sembed, pw)
            for pred_unary, embeddings in self.batch_voxels
        ]
        f_score = self.eval_fn(batch_map, self.batch_gt_labels, self.f_score_beta)
        logging.info(f_score)
        return f_score, (sxyz, sembed, pw)
