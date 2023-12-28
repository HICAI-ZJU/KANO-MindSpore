import math
from argparse import Namespace

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.dtype import List
from mindspore.train import Model
from chemprop_ms.nn_utils import get_activation_function, initialize_weights
from .cmpn import CMPN
from .mpn import MPN
import numpy as np

class MoleculeModel(nn.Cell):
    def __init__(self, classification: bool, multiclass: bool, pretrain: bool):
        super(MoleculeModel, self).__init__(auto_prefix=True)

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(axis=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

    def create_encoder(self, args: Namespace, encoder_name):

        if encoder_name == 'CMPNN':
            self.encoder = CMPN(args)
        elif encoder_name == 'MPNN':
            self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout_3 = nn.Dropout(1- args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = nn.CellList(
                dropout_3,
                nn.Dense(first_linear_dim, args.output_size)
            )
        else:
            ffn = nn.CellList(
                dropout_3,
                nn.Dense(first_linear_dim, args.ffn_hidden_size)
            )
            '''ffn = nn.CellList[
                dropout_3,
                nn.Dense(first_linear_dim, args.ffn_hidden_size)
            ]'''
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout_3,
                    nn.Dense(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout_3,
                nn.Dense(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.SequentialCell(*ffn)

    def construct(self, step, prompt, batch, features_batch):
        """
        Runs the MoleculeModel on input.
        """
        if not self.pretrain:
            output = self.ffn(self.encoder.construct(step = step, prompt = prompt, batch = batch, features_batch= features_batch))

            # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
            if self.classification and not self.training:
                output = self.sigmoid(output)
            if self.multiclass:
                output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
                if not self.training:
                    output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        else:
            output = self.encoder.construct(step = step, prompt = prompt, batch = batch, features_batch= features_batch)

        return output

def build_model(args: Namespace, encoder_name):

    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                            multiclass=args.dataset_type == 'multiclass', pretrain=False)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model


def build_pretrain_model(args: Namespace, encoder_name):
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.
    """
    args.ffn_hidden_size = args.hidden_size // 2
    args.output_size = args.hidden_size

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass', pretrain=True)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model



