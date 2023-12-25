import math

import torch
import torch.nn as nn

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SummaryBothAttention(nn.Module):

    def __init__(self, dim):
        super(SummaryBothAttention, self).__init__()

        self.dim = dim

        self.linear_query1 = nn.Linear(dim, dim, bias=True)
        self.linear_query2 = nn.Linear(dim, dim, bias=True)
        self.linear_cate = nn.Linear(2*dim, 2, bias=True)
        self.v = nn.Linear(dim, dim, bias=False)

    def score(self, h_t1, h_t2, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()  # memory 
        tgt_batch, tgt_len, tgt_dim = h_t1.size()  # query 

        assert src_batch == tgt_batch

        h_t1 = h_t1.view(-1, tgt_dim).unsqueeze(1) # tgt_batch * tgt_len, 1 ,tgt_dim
        h_t2 = h_t2.view(-1, tgt_dim).unsqueeze(1)
        # src_batch * tgt_len, src_len, src_dim
        h_s = h_s.unsqueeze(1).expand(-1,tgt_len,-1,-1).reshape(-1, src_len ,src_dim).contiguous()

        # gate function
        h_t1 = self.linear_query1(h_t1)
        h_t2 = self.linear_query2(h_t2)
        gate_h = self.v(torch.tanh(h_t1 + h_t2 + h_s)) # broadcast to "src_batch * tgt_len, src_len, src_dim"
        sum_gate = torch.sum(gate_h, dim=2) # src_batch * tgt_len, src_len

        h_fuse = self.linear_cate(torch.cat([h_s,sum_gate.unsqueeze(-1)*h_s],-1))
    
        # return F.softmax(h_fuse, -1)
        return h_fuse

    def forward(self, source1, source2, memory_bank, source_masks1=None, source_masks2=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source1.size()
    
        # user和agent有几个seg应该也要一致：(dataloader处加的限制)
        assert source1.size(1) == source2.size(1)

        # 对 query vector 进行处理
        source1 = self.accumulate(source1, source_masks1)
        source2 = self.accumulate(source2, source_masks2)
        # compute attention scores, as in Luong et al.

        # batch*target_l, source_l, 2
        sentence_cate = self.score(source1, source2, memory_bank)
        sentence_cate = sentence_cate.reshape(batch,target_l,source_l,-1)

        return sentence_cate
    
    def accumulate(self, vec, q_mask):
        '''
        Accumulate the sent vector and Mean it 
        '''
        seg_num = vec.size(1)
        batch_sequences = [
                batch_sequence
                for idx, batch_sequence in enumerate(vec)
                ]
        # vec: B,TGT_SEG_NUM,H
        query_list = []
        for sequences in batch_sequences: 
            source = sequences.data.to(sequences)
            for idx in range(1, seg_num):
                # index = torch.arange(idx,0,-1)
                index = torch.tensor(idx*[idx]).to(sequences).int()
                # sequences.index_add_(0, index, source)
                sequences = torch.index_add(sequences, 0, index, source[:index.size(0)])
                sequences[idx] = sequences[idx] / torch.tensor(index.size(0)+1)

            query_list.append(sequences)

        sents_vec = torch.stack(query_list, dim=0)
        sents_vec = sents_vec * q_mask[:, :, None].float()

        return sents_vec


class SummaryAttention(nn.Module):

    def __init__(self, dim):
        super(SummaryAttention, self).__init__()

        self.dim = dim

        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.linear_cate = nn.Linear(2*dim, 2, bias=True)
        self.v = nn.Linear(dim, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()  # memory 
        tgt_batch, tgt_len, tgt_dim = h_t.size()  # query 

        assert src_batch == tgt_batch

        h_t = h_t.view(-1, tgt_dim).unsqueeze(1) # tgt_batch * tgt_len, 1 ,tgt_dim
        # src_batch * tgt_len, src_len, src_dim
        h_s = h_s.unsqueeze(1).expand(-1,tgt_len,-1,-1).reshape(-1, src_len ,src_dim).contiguous()

        # gate function
        h_t = self.linear_query(h_t)
        gate_h = self.v(torch.tanh(h_t + h_s)) # broadcast to "src_batch * tgt_len, src_len, src_dim"
        sum_gate = torch.sum(gate_h, dim=2) # src_batch * tgt_len, src_len

        h_fuse = self.linear_cate(torch.cat([h_s,sum_gate.unsqueeze(-1)*h_s],-1))
    
        # return F.softmax(h_fuse, -1)
        return h_fuse

    def forward(self, source, memory_bank, memory_masks=None, source_masks=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()

        # 对 query vector 进行处理
        source = self.accumulate(source, source_masks)
        # compute attention scores, as in Luong et al.

        # batch*target_l, source_l, 2
        sentence_cate = self.score(source, memory_bank)
        sentence_cate = sentence_cate.reshape(batch,target_l,source_l,-1)

        return sentence_cate
    
    def accumulate(self, vec, q_mask):
        '''
        Accumulate the sent vector and Mean it 
        '''
        seg_num = vec.size(1)
        batch_sequences = [
                batch_sequence
                for idx, batch_sequence in enumerate(vec)
                ]
        # vec: B,TGT_SEG_NUM,H
        query_list = []
        for sequences in batch_sequences: 
            source = sequences.data.to(sequences)
            for idx in range(1, seg_num):
                # index = torch.arange(idx,0,-1)
                index = torch.tensor(idx*[idx]).to(sequences).int()
                # sequences.index_add_(0, index, source)
                sequences = torch.index_add(sequences, 0, index, source[:index.size(0)])
                sequences[idx] = sequences[idx] / torch.tensor(index.size(0)+1)

            query_list.append(sequences)

        sents_vec = torch.stack(query_list, dim=0)
        sents_vec = sents_vec * q_mask[:, :, None].float()

        return sents_vec


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim,  attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)


    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, memory_masks=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()

        # compute attention scores, as in Luong et al.
        # (tgt_batch, tgt_len, src_len)
        align = self.score(source, memory_bank)

        if memory_masks is not None:
            memory_masks = memory_masks.transpose(0,1)
            memory_masks = memory_masks.transpose(1,2)
            align.masked_fill_(1 - memory_masks.byte(), -float('inf'))

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)


        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None,intra_tem=False):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
            elif type == 'role':
                query = self.linear_query(query)
                key = shape(self.linear_keys(key))
                value = shape(self.linear_values(value))

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["inter_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["inter_keys"].to(device), key),
                            dim=2)
                    if layer_cache["inter_values"] is not None:
                        value = torch.cat(
                            (layer_cache["inter_values"].to(device), value),
                            dim=2)
                    layer_cache["inter_keys"] = key
                    layer_cache["inter_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # B,HEAD,tgt_len_user, src_len
        scores = torch.matmul(query, key.transpose(2, 3))

        if intra_tem:
            batch, head, tgt_len, src_len = scores.size()
            scores = scores.reshape(-1, tgt_len, src_len)
            tgt_sum_vec = torch.zeros(scores.size(0),src_len).fill_(1e-10).to(scores)
            tgt_sum_list = []

            # tgt维度进行累加：
            # b*h, tgt, src
            et = torch.exp(scores) # 目的： 防止in-place操作
            scores_ex = et.clone()
            for tgt_idx in range(scores.size(1)):
                # 指数运算
                # if tgt_idx != 0:
                #     scores_ex[:,tgt_idx,:] = et[:,tgt_idx,:]/tgt_sum_vec
                tgt_sum_list.append(tgt_sum_vec)
                # b, src
                tgt_sum_vec = tgt_sum_vec + et[:,tgt_idx,:]
            # b, tgt, src
            tgt_sum_v = torch.stack(tgt_sum_list,1)
            scores_ex[:,1:,:] = et[:,1:,:]/tgt_sum_v[:,1:,:]

            scores_ex = scores_ex.reshape(batch, head, tgt_len, src_len)

        # 此处需要修改：src_batch, tgt_len_user, src_len
        if mask is not None:
            if intra_tem:
                mask = mask.unsqueeze(1).expand_as(scores_ex)
                scores_ex = scores_ex * (~mask)
            else:
                mask = mask.unsqueeze(1).expand_as(scores)
                scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        # TODO: 使用intra_tem 则不用softmax因为已经做了指数运算
        if intra_tem:
            normalization_factor = scores_ex.sum(-1, keepdim=True)
            attn = scores_ex / normalization_factor 
        else:
            attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, scores
        else:
            context = torch.matmul(drop_attn, value)
            return context, scores

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn



class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()
