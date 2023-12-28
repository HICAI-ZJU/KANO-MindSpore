from typing import List

import numpy as np
import mindspore as ms
import mindspore.nn as nn

from chemprop_ms.data import MoleculeDataset, StandardScaler


def predict(model: nn.Cell,
            prompt: bool,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    global batch_preds
    model.set_train(False)

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        step = 'finetune'

        batch = np.array(batch)
        batch = [ms.Tensor(batch)]
        batch_preds = model(step, prompt, batch, features_batch)
        batch_preds = batch_preds.asnumpy()
            #batch_preds = batch_preds.data.numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds


def get_emb(model: nn.Cell,
            prompt: bool,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        step = 'pretrain'

        batch_embs = model.encoder(step, prompt, batch, features_batch)

        batch_embs = batch_embs.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_embs = scaler.inverse_transform(batch_embs)
        
        # Collect vectors
        # batch_embs = batch_embs.tolist()
        # embs.extend(batch_embs)
        if i == 0:
            embs = batch_embs
        else:
            embs = np.vstack((embs, batch_embs)) 

    return embs