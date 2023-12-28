import logging

import mindspore
import mindspore.nn as nn

logger = logging.getLogger()

class NCESoftmaxLoss(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def construct(self, similarity):
        batch_size = similarity.shape[0] // 2
        label = mindspore.Tensor([(batch_size + i) % (batch_size * 2) for i in range(batch_size * 2)],dtype=mindspore.int32)
        loss = self.criterion(similarity, label)
        return loss


class FlatNCE(nn.Cell):
    def __init__(self, device) -> None:
        super().__init__()
        #self.device = device

    def construct(self, similarity):
        pass