import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self, model):
        self.model = model
        self.meters = {
            "loss": AverageMeter(),
            "correct_next": AverageMeter(),
            "correct_next_10": AverageMeter(),
            "correct_any": AverageMeter(),
            "best_rank": AverageMeter(),
            "avg_rank": AverageMeter(),
        }

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, worlds, macros, positives, negatives):
        batch_size = len(worlds)
        outputs = self.model(worlds, macros)
        total_loss, _ = self.model.loss(worlds, outputs, macros)
        self.meters["loss"].update(total_loss.item())

        predictions = self.model.predict(outputs)
        for n in range(batch_size):
            matches = [predictions[n] == positive for positive in positives[n]]
            self.meters["correct_next"].update(matches[0])
            self.meters["correct_next_10"].update(any(matches[:10]))
            self.meters["correct_any"].update(any(matches))

        positive_scores = self.model.score_examples(outputs, positives)
        negative_scores = self.model.score_examples(outputs, negatives)
        ranks = [
            sum(pos_score < neg_score for neg_score in negative_scores)
            for pos_score in positive_scores
        ]
        self.meters["best_rank"].update(min(ranks))
        self.meters["avg_rank"].update(np.mean(ranks), len(ranks))

    def __str__(self):
        return " ".join([f"{name}: {meter.avg:.4f}" for name, meter in self.meters.items()])
