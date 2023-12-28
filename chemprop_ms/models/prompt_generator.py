import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

def attention(query, key, value, mask, dropout=None):
    d_k = ops.shape(query)[-1]
    key_t = key.transpose(1,0)
    scores = ops.matmul(query,key_t) / math.sqrt(d_k)
    if mask is not None:
        mask = ms.Tensor(shape = scores.shape, init = False, dtype = ms.bool_)
        scores = ops.masked_fill(scores, mask, -1e9)
    p_attn = ops.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return ops.matmul(p_attn, value), p_attn

class AttentionLayer(nn.Cell):
    def __init__(self, args):
        super(AttentionLayer, self).__init__(auto_prefix=True)
        self.hidden_size = args.hidden_size
        self.w_q = nn.Dense(133, 32)
        self.w_k = nn.Dense(133, 32)
        self.w_v = nn.Dense(133, 32)

        self.dense = nn.Dense(32, 133)
        self.LayerNorm = nn.LayerNorm([133], epsilon=1e-6)
        self.dropout = nn.Dropout(0.9)

    def construct(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = ops.matmul(padding_mask, padding_mask.transpose(1, 0))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)

        return hidden_states

class Prompt_generator(nn.Cell):
    def __init__(self, args):
        super(Prompt_generator, self).__init__(auto_prefix=True)
        self.hidden_size = args.hidden_size
        self.alpha = ms.Parameter(ms.Tensor(0.1, ms.float32), requires_grad=True)
        #self.alpha.data.fill_(0.1)
        self.cls = ms.Parameter(ops.randn(1,133), requires_grad=True, name="cls_3")
        self.linear = nn.Dense(133, self.hidden_size)
        # self.attention_layer = nn.MultiheadAttention(133, 1, 0.9, batch_first=True)
        # self.w_q = nn.Dense(133, 32)
        # self.w_k = nn.Dense(133, 32)
        # self.w_v = nn.Dense(133, 32)
        self.attention_layer_1 = AttentionLayer(args)
        self.attention_layer_2 = AttentionLayer(args)
        self.norm = nn.LayerNorm([args.hidden_size])

    def construct(self, atom_hiddens: ms.Tensor, fg_states: ms.Tensor, atom_num, fg_indexs):
        for i in range(len(fg_indexs)):
            fg_states = ops.scatter(fg_states, 0, fg_indexs[i:i+1], self.cls)

        hidden_states = self.attention_layer_1.construct(fg_states, fg_states)
        hidden_states = self.attention_layer_2.construct(hidden_states, fg_states)
        # query = self.w_q(fg_states)
        # key = self.w_k(fg_states)
        # value = self.w_v(fg_states)
        # padding_mask = (fg_states != 0) + 0.0
        # mask = ops.matmul(padding_mask, padding_mask.transpose(1, 0))
        # hidden_states, _ = self.attention_layer(query, key, value, attn_mask=mask)
        fg_out = ops.zeros((1, self.hidden_size))
        cls_hiddens = ops.gather_elements(hidden_states, 0, fg_indexs)
        cls_hiddens = self.linear(cls_hiddens)
        fg_hiddens = ms.Tensor(np.repeat(cls_hiddens.asnumpy(), np.array(atom_num), axis=0))
        fg_out = ops.cat((fg_out, fg_hiddens), 0)
        fg_out = self.norm(fg_out)
        return atom_hiddens + self.alpha * fg_out


class PromptGeneratorOutput(nn.Cell):
    def __init__(self, args, self_output):
        super(PromptGeneratorOutput, self).__init__(auto_prefix=True)
        # change position
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)

    def construct(self, hidden_states: ms.Tensor):
        hidden_states = self.self_out(hidden_states)
        return hidden_states


def prompt_generator_output(args):
    return lambda self_output: PromptGeneratorOutput(args, self_output)


def add_functional_prompt(model, args):
    model.encoder.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.encoder.W_i_atom)
    return model