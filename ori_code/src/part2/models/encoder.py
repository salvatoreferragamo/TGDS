import math

import torch
import torch.nn as nn
import numpy as np

from models.neural import MultiHeadedAttention, PositionwiseFeedForward


MAX_SIZE = 5000
class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        if mask.dim() == 2: 
            mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers, label_class):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.wo = nn.Linear(d_model, 1, bias=True)
        # bio
        #self.wo = nn.Linear(d_model, 5, bias=True)
        self.wo = nn.Linear(d_model, label_class, bias=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        # 句子及间的注意力机制
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        # sent_scores = self.sigmoid(self.wo(x))
        # sent_scores = sent_scores.squeeze(-1) * mask.float()
        # bio
        sent_scores = self.wo(x)
        # [B,cls,class]
        sent_scores = sent_scores.squeeze(-1) * mask.float().unsqueeze(-1)

        return sent_scores


class TgtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers, embeddings):
        super(TgtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, seg_idx, seg_mask, tgt_mask, step=None):
        """ See :obj:`EncoderBase.forward()`  
        seg_idx  B,ID_LEN
        tgt_mask B,TGT_LEN
        """
        tgt_batch, tgt_len = tgt_mask.size()
        emb = self.embeddings(tgt)
        # n_sents = src.size(1)
        # pos_emb = self.pos_emb.pe[:, :n_sents]
        # x = emb * mask[:, :, None].float()
        # x = x + pos_emb
        assert emb.dim() == 3  # len x batch x embedding_dim
        x = self.pos_emb(emb, step)
        # 句子及间的注意力机制(mask 需要修改成为cross形式)
        mask = self.construct_mask(seg_idx, seg_mask, MAX_SIZE)
        tgt_pad_mask = tgt_mask.data.eq(0).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        # mask掉的部分是True , 注意
        mask = torch.gt(tgt_pad_mask +
                            mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)].to(tgt_pad_mask), 0)

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, mask)  # all_sents * max_tokens * dim
        
        x = self.layer_norm(x)

        return x

    def construct_mask(self, seg_idx, seg_mask, size):
        """
        construct an attention mask to avoid using the tgt subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (seg_idx.size(0), size, size)
        subsequent_mask = np.ones(attn_shape)
        for i in range(seg_idx.size(0)):
            subsequent_mask[i][0][0] = 0
            # print(seg_idx[i],seg_mask[i])
            for j in range(int(torch.sum(seg_mask[i]))):
                if j != (int(torch.sum(seg_mask[i]))-1):
                    # print(j)
                    subsequent_mask[i][int(seg_idx[i][j])+1:int(seg_idx[i][j+1])+1][:int(seg_idx[i][j+1])+1] = 0
                else: 
                    subsequent_mask[i][int(seg_idx[i][j])+1:][:] = 0

        subsequent_mask = torch.from_numpy(subsequent_mask.astype('uint8'))

        return subsequent_mask


class ContextTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers):
        super(ContextTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def _get_attn_subsequent_mask(self, size, k=1):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=k).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

    def forward(self, top_vecs, mask, is_cross=False):
        """ See :obj:`EncoderBase.forward()`"""

        # utterances num or seg nums
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        # print(top_vecs.size(),mask.size())
        pos_emb = self.pos_emb.pe[:, :n_sents]
        # mask 
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        # 句子及间的注意力机制(tgt的句子间要用cross attention)
        if not is_cross:
            for i in range(self.num_inter_layers):
                x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
        else:
            cross_mask = self._get_attn_subsequent_mask(MAX_SIZE)
            mask = ~mask.unsqueeze(1).expand(batch_size, n_sents, n_sents)
            dec_mask = torch.gt(mask +
                            cross_mask[:, :mask.size(1),
                            :mask.size(1)].to(mask), 0)
            for i in range(self.num_inter_layers):
                x = self.transformer_inter[i](i, x, x, dec_mask) 

        x = self.layer_norm(x)

        return x
    