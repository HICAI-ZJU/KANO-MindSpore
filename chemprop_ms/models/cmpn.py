import math
from argparse import Namespace
from typing import List

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from chemprop_ms.nn_utils import index_select_ND, get_activation_function
from chemprop_ms.features.featurization import get_atom_fdim, get_bond_fdim, mol2graph
from chemprop_ms.models.prompt_generator import Prompt_generator
import numpy

class CMPNEncoder(nn.Cell):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__(auto_prefix=True)
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.dropout = args.dropout
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.args = args

        self.dropout_layer = nn.Dropout(keep_prob =1 - self.dropout)
        self.prompt = Prompt_generator(self.args)
        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Dense(input_dim, self.hidden_size, bias_init=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Dense(input_dim, self.hidden_size, bias_init=self.bias)

        self.act_func = get_activation_function(args.activation)

        # add & concat functional group features
        self.cls = ms.Parameter(ops.randn(1, 133), name ="cls_cmpn")
        self.W_i_atom_new = nn.Dense(self.atom_fdim * 2, self.hidden_size, bias_init=self.bias)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Dense(self.hidden_size * 3, self.hidden_size, bias_init=self.bias)

        self.W_o = nn.Dense((self.hidden_size) * 2, self.hidden_size)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom_1 = nn.Dense(w_h_input_size_atom, self.hidden_size, bias_init=self.bias)

        w_h_input_size_bond = self.hidden_size
        for depth in range(self.depth-1):
            self._cells[f'W_h_{depth}'] = nn.Dense(w_h_input_size_bond, self.hidden_size, bias_init=self.bias)

    def construct(self, step, mol_graph, features_batch=None ):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num, fg_num, f_fgs, fg_scope = mol_graph.get_components()
        '''if self.args.cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, f_fgs = (
                    f_atoms.cuda(), f_bonds.cuda(),
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(), f_fgs.cuda())'''

        fg_index = [i * 13 for i in range(mol_graph.n_mols)]
        fg_indxs = [[i] * 133 for i in fg_index]
        fg_indxs = ms.Tensor(fg_indxs, ms.int32)

        if self.args.step == 'functional_prompt':
            # make sure the prompt exists
            #assert self.W_i_atom_1.prompt_generator
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            input_atom = self.prompt.construct(atom_hiddens = input_atom, fg_states = f_fgs,atom_num = atom_num, fg_indexs = fg_indxs)


        elif self.args.step == 'finetune_concat':
            for i in range(len(fg_indxs)):
                #f_fgs.ops.scatter_(0, fg_indxs[i:i + 1], self.cls)
                f_fgs = ops.scatter(f_fgs, 0, fg_indxs[i:i + 1], self.cls)

            #target_index = [val for val in range(mol_graph.n_mols) for _ in range(13)]
            target_index=[]
            for val in range(mol_graph.n_mols):
                for i in range(13):
                    target_index = target_index.append(val)
            target_index = ms.Tensor(target_index, ms.int64)
            zero_tensor = ops.Zeros()(target_index, ms.int64)
            fg_hiddens = ops.tensor_scatter_elements(f_fgs, target_index ,zero_tensor)
            fg_hiddens_atom = ops.repeat_interleave(fg_hiddens, ms.Tensor(atom_num), axis=0)
            fg_out = ops.zeros(1, 133)
            fg_out = ops.cat((fg_out, fg_hiddens_atom), 0)
            f_atoms = ops.cat((fg_out, f_atoms), 1)
            # Input
            input_atom = self.W_i_atom_new(f_atoms)

        else:
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size

        input_atom = self.act_func(input_atom)
        message_atom = input_atom.copy()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(axis=1) * agg_message.max(axis=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._cells[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(axis=1) * agg_message.max(axis=1)[0]
        agg_message = self.lr(ops.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))

        mol_vecs = ops.stack(mol_vecs, 0)
        return mol_vecs  # B x H

class BatchGRU(nn.Cell):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__(auto_prefix=True)
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                            bidirectional=True)
        self.bias = ms.Parameter(ops.uniform(shape = ms.Tensor(self.hidden_size).shape, minval = ms.Tensor(-1.0 / math.sqrt(self.hidden_size)),
                                             maxval = ms.Tensor(1.0 / math.sqrt(self.hidden_size))))


    def construct(self, node, a_scope):
        hidden = node
        message = ops.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            x= cur_hidden.max(axis = 0)
            hidden_lst.append(ops.unsqueeze(ops.unsqueeze(x,0),0))

            cur_message = nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(ops.unsqueeze(cur_message,0))

        message_lst = ops.cat(message_lst, 0)
        hidden_lst = ops.cat(hidden_lst, 1)
        hidden_lst = ops.tile(hidden_lst,(2, 1, 1))
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = ops.cat(cur_message_unpadding, 0)

        message = ops.cat([ops.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


class CMPN(nn.Cell):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__(auto_prefix=True)
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                            (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def construct(self, step, prompt: bool, batch,
                features_batch: List[np.ndarray] = None):
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args, prompt)

        output = self.encoder.construct(step = step, mol_graph = batch)
        return output
