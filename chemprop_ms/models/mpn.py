from argparse import Namespace
from typing import List

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from ..features import get_atom_fdim, get_bond_fdim, mol2graph
from ..nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Cell):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__(auto_prefix=True)
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.dropout = args.dropout
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.rtype = 'mean'
        self.aggregation_norm = 100

        self.dropout_layer_2 = nn.Dropout(keep_prob=1 - self.dropout)

        # Cached zeros
        self.cached_zero_vector = ms.Parameter(ops.Zeros()(self.hidden_size, ms.float32),name="cached_zero_vector_1")

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i_2 = nn.Dense(input_dim, self.hidden_size, bias_init=self.bias)


        self.act_func = get_activation_function(args.activation)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h_2 = nn.Dense(w_h_input_size, self.hidden_size, bias_init=self.bias)

        self.W_o_2 = nn.Dense(self.atom_fdim + self.hidden_size, self.hidden_size)

    def construct(self, mol_graph , ):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, _ = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        if self.atom_messages:
            input = self.W_i_2(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i_2(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().cuda()

        '''
        #dynamic depth
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = numpy.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth
        '''
        '''
        if self.undirected:
            message = (message + message[b2revb]) / 2
        '''
        for depth in range(self.depth - 1):

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = ops.cat((nei_a_message, nei_f_bonds),1)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                '''
                if f_bonds：
                    atom_rev_message = f_bonds[b2a[b2revb]]
                    rev_message = ops.concat((rev_message, atom_rev_message), 1)
                '''
                message = a_message[b2a] - rev_message

                message = self.W_h_2(message)
                message = self.act_func(input + message)
                message = self.dropout_layer_2(message)

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)
        a_input = ops.cat([f_atoms, a_message], 1)
        atom_hiddens = self.act_func(self.W_o_2(a_input))
        atom_hiddens = self.dropout_layer_2(atom_hiddens)

        # Readout
        mol_vecs = []
        for (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                if self.rtype == 'mean':
                    cur_hiddens = cur_hiddens.sum(0) / a_size
                elif self.aggregation == 'sum':
                    cur_hiddens = cur_hiddens.sum(0)
                elif self.aggregation == 'norm':
                    cur_hiddens = cur_hiddens.sum(0) / self.aggregation_norm

                mol_vecs.append(cur_hiddens)

        mol_vecs = ops.stack(mol_vecs, 0)
        return mol_vecs


class MPN(nn.Cell):

    def __init__(self, args: Namespace, atom_fdim: int = None, bond_fdim: int = None):
        super().__init__(auto_prefix=True)
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)

    def construct(self, prompt: bool,
                batch,
                features_batch: List[np.ndarray] = None):
        batch = mol2graph(batch, self.args, prompt)
        output = self.encoder.construct(batch)
        return output




