"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        # diff head 默认false
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout, diff_head=True)
        # self.context_attn = MultiHeadedAttention(
        #     heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)
        self.mid_gate = nn.Linear(2 * d_model, 1, bias=False)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, role_mask, src_tgt_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        # tgt zero padding 和 casual padding
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        # inference 用到：
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        # self attention :
        query,_ = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        # cross attention:
        #TODO: 此处加上分别处理不同role的context cross att
        # b, tgt, src (ne 不等于)
        # print(src_pad_mask.size())
        # print(role_mask.size())
        assert src_tgt_mask.size(1) == src_pad_mask.size(1)

        src_pad_mask_user = src_pad_mask | role_mask.ne(1).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        src_pad_mask_agent = src_pad_mask | role_mask.ne(2).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        # 增加的utt mask:
        src_pad_mask_user_ = src_tgt_mask.to(src_pad_mask) | src_pad_mask | role_mask.ne(1).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        src_pad_mask_agent_ = src_tgt_mask.to(src_pad_mask) | src_pad_mask | role_mask.ne(2).unsqueeze(1).expand(role_mask.shape[0], src_pad_mask.shape[1], role_mask.shape[1])
        # mid,_ = self.context_attn(memory_bank, memory_bank, query_norm,
        #                               mask=src_pad_mask,
        #                               layer_cache=layer_cache,
        #                               type="context")
        # TODO: real utterance mask (multi-head)
        # b, tgt, h
        
        mid_user, user_score = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask_user,
                                      add_mask=src_pad_mask_user_,
                                      layer_cache=layer_cache,
                                      type="context")
        mid_agent, agent_score= self.context_attn(memory_bank, memory_bank, query_norm,
                                     mask=src_pad_mask_agent,
                                     add_mask=src_pad_mask_agent_,
                                     layer_cache=layer_cache,
                                     type="context")

        # use to allocate diff role weight:
        mid_query = torch.cat([mid_user, mid_agent], dim=-1).view(-1, mid_user.shape[-1] * 2)
        # b*tgt, hidden*2
        mid_gate = nn.Sigmoid()(self.mid_gate(mid_query)).view(mid_user.shape[0], mid_user.shape[1], 1)
        mid = mid_user * mid_gate + mid_agent * (1 - mid_gate)
        # b, h, tgt, src
        mid_gate = mid_gate.unsqueeze(1)
        tog_score = F.softmax(user_score,dim=-1) * mid_gate + F.softmax(agent_score,dim=-1) * (1 - mid_gate)
        
        output = self.feed_forward(self.drop(mid) + query)

        return output, all_input, tog_score
        
        mid, att_score = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        tog_score = F.softmax(att_score,dim=-1)
        output = self.feed_forward(self.drop(mid) + query)
        return output, all_input, tog_score
        return output, all_input, user_score, agent_score, tog_score
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask



class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.dropout_layer = nn.Dropout(0.3)


        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, role_mask, tgt_user_mask=None, tgt_agent_mask=None, src_tgt_mask_tog=None, 
                memory_lengths=None,step=None, cache=None, memory_masks=None, tgt_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        src_tgt_mask_tog : b, all_tgt, src 
        """

        # 一些向量的大小信息：
        src_words = state.src
        batch, src_len = src_words.size()

        tgt_len = tgt.size(1)

        # 进行token emb 与 pos emb 的映射：
        emb_tgt = self.embeddings(tgt)
        assert emb_tgt.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb_tgt, step)

        # 对于tgt mask 以及src mask的处理：
        padding_idx = self.embeddings.padding_idx

        tgt_pad_mask = tgt.data.eq(padding_idx).unsqueeze(1) \
            .expand(batch, tgt_len, tgt_len)
        
        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(batch, tgt_len, src_len)
        else:
            # b, 1, src_len
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(batch, tgt_len, src_len)

        # utt mask:
        ### gold label mask
        src_tgt_mask_tog = src_tgt_mask_tog.eq(padding_idx)
        if src_tgt_mask_tog.dim() == 3:
            src_tgt_mask_tog_ = src_tgt_mask_tog[:,1:,]
        else:
            src_tgt_mask_tog_ = src_tgt_mask_tog.unsqueeze(1)
        # 处理好 decoder的输入， 可以进入transformer layer中进行运算
        src_memory_bank = memory_bank

        # KL loss 计算的 mask 
        # kl_mask = tgt.data.ne(padding_idx).unsqueeze(2).expand(batch, tgt_len, src_len)

        # 存放对应的 att score
        user_scores, agent_scores = [[], []], [[], []]
        last_layer_score =[]
        # 未使用cache保存
        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                # 通过不断更新state的方式来完成层级之间的传递(inference的时候使用)
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            # decoder layer
            # user_score, agent_score :  b, head, tgt_len, src 
            # output, all_input, user_score, agent_score, tog_score \
            #     = self.transformer_layers[i](
            #         output, src_memory_bank,
            #         src_pad_mask, tgt_pad_mask, role_mask, src_tgt_mask_tog_,
            #         previous_input=prev_layer_input,
            #         layer_cache=state.cache["layer_{}".format(i)]
            #         if state.cache is not None else None,
            #         step=step)
            output, all_input, tog_score \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask, role_mask, src_tgt_mask_tog_,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)

            # 根据tgt_user_mask, tgt_agent_mask 选择相应的score vec:
            '''
            if tgt_user_mask is not None:
                head, dim , device = user_score.size(1), user_score.size(-1), user_score.device
                # b*head, tgt, src
                user_score, agent_score = self._bottle(user_score), self._bottle(agent_score)
                assert user_score.dim() == 3

                u_u_scores_list, u_a_scores_list = [], []
                a_u_scores_list, a_a_scores_list = [], []
                for i in range(user_score.size(0)):
                    # list: user_len,d || agent_len,d
                    # broadcast mask user or agent:
                    # u_u_scores_list.append(torch.masked_select(user_score[i], tgt_user_mask[i//head].unsqueeze(-1)).reshape(-1,dim))
                    # a_u_scores_list.append(torch.masked_select(user_score[i], tgt_agent_mask[i//head].unsqueeze(-1)).reshape(-1,dim))
                    # u_a_scores_list.append(torch.masked_select(agent_score[i], tgt_user_mask[i//head].unsqueeze(-1)).reshape(-1,dim))
                    # a_a_scores_list.append(torch.masked_select(agent_score[i], tgt_agent_mask[i//head].unsqueeze(-1)).reshape(-1,dim))
                    u_u_scores_list.append(user_score[i].masked_fill(tgt_user_mask[i//head].unsqueeze(-1),-1e32))
                    a_u_scores_list.append(user_score[i].masked_fill(tgt_agent_mask[i//head].unsqueeze(-1),-1e32))
                    u_a_scores_list.append(agent_score[i].masked_fill(tgt_user_mask[i//head].unsqueeze(-1),-1e32))
                    a_a_scores_list.append(agent_score[i].masked_fill(tgt_agent_mask[i//head].unsqueeze(-1),-1e32))

                # b, head, user/agent_len, src
                u_u_vec = self._unbottle(torch.stack(u_u_scores_list).to(device),head)
                a_u_vec = self._unbottle(torch.stack(a_u_scores_list).to(device),head)
                u_a_vec = self._unbottle(torch.stack(u_a_scores_list).to(device),head)
                a_a_vec = self._unbottle(torch.stack(a_a_scores_list).to(device),head)

                # 将score保存起来方便后面计算loss:
                # 每一层的 b, user/agent_len, src
                user_scores[0].append(torch.mean(u_u_vec, dim=1))
                agent_scores[0].append(torch.mean(u_a_vec, dim=1))
                user_scores[1].append(torch.mean(a_u_vec, dim=1))
                agent_scores[1].append(torch.mean(a_a_vec, dim=1))
            '''

            if state.cache is None:
                saved_inputs.append(all_input)
        '''
        # 最后一层的 user_score, agent_score 保存起来用于PGN：
        # b,head,tgt,src -> b,tgt,src
        if tgt_user_mask is not None:
            user_score, agent_score = self._unbottle(user_score,head), self._unbottle(agent_score,head)
        # last_layer_score.extend([torch.mean(user_score, dim=1),torch.mean(agent_score, dim=1)])
        last_layer_score.append(torch.mean(tog_score, dim=1))
        '''
        # b,tgt,src
        last_layer_score.append(torch.mean(tog_score, dim=1))

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)
        output = self.dropout_layer(output)

        # Process the result and update the attentions.

        if state.cache is None:
            # 对previous_input和previous_layer_inputs 进行更新
            state = state.update_state(tgt, saved_inputs)

        return output, state, last_layer_score
        return output, state, user_scores, agent_scores, last_layer_score

    # 初始化TransformerDecoderState
    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            # 初始化state.cache
            state._init_cache(memory_bank, self.num_layers)
        return state

    def _bottle(self, _v):
        return _v.reshape(-1, _v.size(2), _v.size(-1))

    def _unbottle(self, _v, head):
        return _v.reshape(-1, head, _v.size(1), _v.size(-1))



class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()
    
    # 对previous_input和previous_layer_inputs 进行更新
    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        # 对cache进行一些初始化
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    #TODO: 后两个函数的作用：？(在inference阶段进行)

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    # 对src以及所有的keys values 进行tile操作 B -> B*beam_size
    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)



