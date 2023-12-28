import logging

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .loss_computer import NCESoftmaxLoss

logger = logging.getLogger()

class ContrastiveLoss(nn.Cell):
    def __init__(self, loss_computer: str, temperature: float, args) -> None:
        super().__init__()
        #self.device = args.device

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss()
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    def construct(self, z_i, z_j):
        batch_size = ops.shape(z_i)[0]
        l2_normalize = ops.L2Normalize()
        emb = l2_normalize(ops.cat([z_i, z_j]))
        #emb = ops.L2Normalize(ops.cat([z_i, z_j]))

        similarity = ops.matmul(emb, emb.t()) - ops.eye(n=batch_size * 2,m =batch_size * 2 ,dtype=ms.float32) * 1e12
        similarity = similarity * 20
        loss = self.loss_computer(similarity)

        return loss


class FlatNCE(nn.Cell):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()

    def construct(self, z_i, z_j):
        def __init__(self, temperature):
            self.temperature = temperature
            super().__init__()

        def construct(self, z_i, z_j):
            batch_size = z_i.size(0)

            features = ops.cat([z_i, z_j], axis=0)
            labels = ops.cat([ops.arange(batch_size) for i in range(2)], axis=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            features = ops.L2Normalize(features, axis=1)

            similarity_matrix = ops.matmul(features, features.T)

            mask = ops.eye(labels.shape[0], dtype=ms.bool_)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

            # logits = torch.cat([positives, negatives], dim=1)
            labels = ops.zeros(positives.shape[0], dtype=ms.float32)
            logits = (negatives - positives) / self.temperature
            clogits = ops.logsumexp(logits, axis=1, keep_dims=True)
            loss = ops.exp(clogits - clogits.detach())



