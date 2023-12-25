import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder, RoleDecoder
import models.decoder_all as decoder_all
from models.encoder import Classifier, ExtTransformerEncoder, TgtTransformerEncoder, ContextTransformerEncoder
# from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from crf import CRFLayer
from models.pooling import Pooling
from models.neural import SummaryAttention, SummaryBothAttention

from others.utils import rouge_results_to_str, test_rouge, tile

def build_optim(args, model, checkpoint, gen_op):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][2]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)
    
    params = [(n, p) for n, p in list(model.named_parameters()) if n in gen_op]
    optim.set_parameters(params)


    return optim

def build_optim_bert(args, model, checkpoint, bert_op):
    """ Build optimizer """

    if checkpoint is not None:
        # KL loss finetine 需要变换
        optim = checkpoint['optims'][0]
        # optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    # params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model') or n.startswith('user_tgt_layer')
    #             or n.startswith('agent_tgt_layer')]
    params = [(n, p) for n, p in list(model.named_parameters()) if n in bert_op]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint, bert_op):
    """ Build optimizer """

    # 修改一下判断条件
    # if checkpoint is None:
    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    # params = [(n, p) for n, p in list(model.named_parameters()) if not (n.startswith('bert.model') and n.startswith('user_tgt_layer')
    #             and n.startswith('agent_tgt_layer'))]
    params = [(n, p) for n, p in list(model.named_parameters()) if n not in bert_op]
    optim.set_parameters(params)


    return optim

# 映射到vocab中去（这块应该可以进行改进）
def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

# 生成器 为PGN：
class CopyGenerator(nn.Module):
    def __init__(self, vocab_size, dec_hidden_size, device):
        super(CopyGenerator, self).__init__()

        self.p_gen_linear = nn.Linear(dec_hidden_size * 2 , 1)
        # self.p_role_linear = nn.Linear(dec_hidden_size * 3 , 1)

        self.out1 = nn.Linear(dec_hidden_size, dec_hidden_size)
        self.out2 = nn.Linear(dec_hidden_size, vocab_size)
        self.vocab_size = vocab_size

        self.to(device)

    def forward(self, output, last_layer_score, enc_batch_extend_vocab):
        # cat 之后的作为输入
        s_output, state_input = output[0], output[1]
        batch, tgt_len, src_len = enc_batch_extend_vocab.size(0), state_input.size(1), enc_batch_extend_vocab.size(1)

        p_gen = self.p_gen_linear(state_input)
        p_gen = torch.sigmoid(p_gen)
        # p_role = self.p_role_linear(state_input)
        # p_role = torch.sigmoid(p_role)

        output = self.out1(s_output)
        output = self.out2(output)  # B * tgt * vocab_size
        vocab_dist = F.softmax(output, dim=-1)

        # b, tgt, src
        # attn_dist_user, attn_dist_agent = F.softmax(last_layer_score[0], dim=-1), F.softmax(last_layer_score[1], dim=-1)
        # normalization_factor_user, normalization_factor_agent = attn_dist_user.sum(2, keepdim=True),attn_dist_agent.sum(2, keepdim=True)
        # attn_dist_user, attn_dist_agent = attn_dist_user / normalization_factor_user, attn_dist_agent / normalization_factor_agent
        # attn_dist = F.softmax(last_layer_score[0], dim=-1)
        # normalization_factor = attn_dist.sum(2, keepdim=True)
        # attn_dist = attn_dist / normalization_factor

        attn_dist = last_layer_score[0]
        vocab_dist_ = p_gen * vocab_dist #b, tgt, vocab
        # attn_dist = p_role * attn_dist_user + (1 - p_role) * attn_dist_agent
        attn_dist_ = (1 - p_gen) * attn_dist #b, tgt, src

        enc_batch_extend_vocab = enc_batch_extend_vocab.unsqueeze(1).expand(batch, tgt_len, src_len)

        # pointer net:
        # b, tgt, vocab
        final_dist = vocab_dist_.scatter_add(2, enc_batch_extend_vocab, attn_dist_) + 1e-12
        final_dist = final_dist.log()

        return final_dist.reshape(-1,self.vocab_size)


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('../pretrained/bert_base_chinese', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('../pretrained/bert_base_chinese', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            # top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
            # 输出格式变了，只需要last_hidden_states
            top_vec = self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state
            # print(self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state.size())
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state
                # top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        # self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
        #                                        args.ext_dropout, args.ext_layers, args.label_class)
        # self.tag_to_ix = {"B": 1, "I": 2, "O": 0, "<START>": 3, "<STOP>": 4}
        # self.crf_layer = CRFLayer(self.tag_to_ix)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        # 在加一个专门用来训练decoder部分gold label的transformer encoder
        self.user_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        self.agent_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        # get sent context info 
        self.src_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, 1)
        self.user_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, 1)
        self.agent_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, 1)
        # 得到句向量池化层：
        self.pooling = Pooling(args.sent_rep_tokens, args.mean_tokens, args.max_tokens)
        # 得分函数：
        # self.user_score_layer = SummaryAttention(self.bert.model.config.hidden_size)
        # self.agent_score_layer = SummaryAttention(self.bert.model.config.hidden_size)

        self.score_layer = SummaryBothAttention(self.bert.model.config.hidden_size)

        self.classifier = nn.Linear(self.bert.model.config.hidden_size, 2, bias=True)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            # if args.param_init != 0.0:
            #     for p in self.ext_layer.parameters():
            #         p.data.uniform_(-args.param_init, args.param_init)
            # if args.param_init_glorot:
            #     for p in self.ext_layer.parameters():
            #         if p.dim() > 1:
            #             xavier_uniform_(p)
            # initialize params.
            if args.encoder == "transformer":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
            for module in self.user_tgt_layer.modules():
                self._set_parameter_tf(module)
            for module in self.agent_tgt_layer.modules():
                self._set_parameter_tf(module)
            for module in self.src_context_layer.modules():
                self._set_parameter_tf(module)
            for module in self.user_tgt_context_layer.modules():
                self._set_parameter_tf(module)
            for module in self.agent_tgt_context_layer.modules():
                self._set_parameter_tf(module)
            for p in self.score_layer.parameters():
                self._set_parameter_linear(p)
            for p in self.classifier.parameters():
                self._set_parameter_linear(p)

            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.user_tgt_layer.embeddings = tgt_embeddings
                self.agent_tgt_layer.embeddings = tgt_embeddings

        self.to(device)

    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    # def forward(self, src, segs, clss, mask_src, mask_cls):
    def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final):
        top_vec = self.bert(src, segs, mask_src)
        # print(top_vec)
        # [B,1]
        # [B,cls_num,H] 
        # clss ： [B,cls_num]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        src_sent_vec = sents_vec * mask_cls[:, :, None].float()
        # # 对cls hidden states进行classify
        # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        # 直接先粗俗的取词向量平均(取max)转化为句向量：

        ########################################################################
        # src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
        # 对golden label 进行编码：
        # tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
        # tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
        # # sentence context 计算：
        src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
        # tgt_user_sent_vec = self.user_tgt_context_layer(tgt_user_sent_vec,mask_tgt_lens[0],True)
        # tgt_agent_sent_vec = self.agent_tgt_context_layer(tgt_agent_sent_vec,mask_tgt_lens[1],True)
        # Score函数(user and agent) batch, target_l, source_l, 2
        # 此处考虑agent和user的合二为一看看怎么样
        #########################################################################

        # user_utt_scores = self.user_score_layer(tgt_user_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[0])
        # agent_utt_scores = self.user_score_layer(tgt_agent_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[1])

        # batch, target_l, source_l, 2
        # scores = self.score_layer(tgt_user_sent_vec,tgt_agent_sent_vec,src_sent_vec,mask_tgt_lens[0],mask_tgt_lens[1])
        #######直接对utterance分类我看一下怎么样效果：(句子分类对应ext任务)
        # B, utterance_num, 2 
        scores = self.classifier(src_sent_vec)
        
        # return [user_utt_scores,agent_utt_scores]
        return scores
        # CRF
        #crf_scores = self.crf_layer.neg_log_likelihood(sent_scores, tgt)
        #return crf_scores, mask_cls

    def decode(self, src, segs, clss, mask_src, mask_cls, tgt):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        tag_seq = self.crf_layer.forward_test(sent_scores)
        return tag_seq


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None, symbols=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        # special id
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        # self.user_tag_id, self.agent_tag_id = symbols['USER'], symbols['AGENT']
        self.user_tag_id, self.agent_tag_id = 1,2

        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            # 打印出来看一看：
            # for n, p in bert_from_extractive.items():
            #     print(n)
            
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        # embedding set 
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        # 在加一个专门用来训练decoder部分gold label的transformer encoder
        self.user_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        self.agent_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        # get sent context info 
        self.src_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, 1)
        self.user_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, 1)
        self.agent_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
                                               args.tgt_dropout, 1)
        # 得到句向量池化层：
        self.pooling = Pooling(args.sent_rep_tokens, args.mean_tokens, args.max_tokens)
        self.pooling1 = Pooling(args.sent_rep_tokens, False, True)
        # 得分函数：
        self.score_layer = SummaryBothAttention(self.bert.model.config.hidden_size)
        # self.user_score_layer = SummaryAttention(self.bert.model.config.hidden_size)
        # self.agent_score_layer = SummaryAttention(self.bert.model.config.hidden_size)

        self.user_decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        self.agent_decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        self.role_decoder = RoleDecoder(self.user_decoder, self.agent_decoder)

        # 一体化解码：
        self.tog_decoder = decoder_all.TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        # todo: different generator? make it an option, first set it to be the same
        # self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator = CopyGenerator(self.vocab_size, self.args.dec_hidden_size, device)
        # self.generator.out2.weight = self.tog_decoder.embeddings.weight

        # 这块不太对，应该来说是要初始化decoder部分，但是之前的ext只有encoder部分
        # train 和 test 要来回改这块，很麻烦
        # if checkpoint is None:
        #     pass
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.user_decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.agent_decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.role_decoder.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.generator.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.user_tgt_layer.modules():
                self._set_parameter_tf(module)
            for module in self.agent_tgt_layer.modules():
                self._set_parameter_tf(module)
            for module in self.src_context_layer.modules():
                self._set_parameter_tf(module)
            for module in self.user_tgt_context_layer.modules():
                self._set_parameter_tf(module)
            for module in self.agent_tgt_context_layer.modules():
                self._set_parameter_tf(module)
            for p in self.score_layer.parameters():
                self._set_parameter_linear(p)
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.user_decoder.embeddings = tgt_embeddings
                self.agent_decoder.embeddings = tgt_embeddings
                self.user_tgt_layer.embeddings = tgt_embeddings
                self.agent_tgt_layer.embeddings = tgt_embeddings
                self.generator.out2.weight = self.tog_decoder.embeddings.weight

        self.to(device)
    
    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()
            
    # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances mask_utterances, src_tgt_mask_final):
    # def forward(self, src, tgts, segs, mask_src, role_mask, tgt_fin_role, src_tgt_mask_final, src_lens, mask_src_lens, tgt_fin_len):
    def forward(self, src, tgts, segs, mask_src, role_mask, tgt_fin_role, src_tgt_mask_final):
    # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, src_tgt_mask_final):
        
        top_vec = self.bert(src, segs, mask_src)
        #### 分别得到src与tgt的encoder vector 
        # 直接先粗俗的取词向量平均(取max)转化为句向量：
        # src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
        
        #################################
        # 对golden label 进行编码：
        # tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
        # tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
        # sentence context 计算：
        # src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
        # tgt_user_sent_vec = self.user_tgt_context_layer(tgt_user_sent_vec,mask_tgt_lens[0],True)
        # tgt_agent_sent_vec = self.agent_tgt_context_layer(tgt_agent_sent_vec,mask_tgt_lens[1],True)
        # # Score函数(user and agent) batch, target_l, source_l, 2
        # # user_utt_scores = self.user_score_layer(tgt_user_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[0])
        # # agent_utt_scores = self.user_score_layer(tgt_agent_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[1])
        # scores = self.score_layer(tgt_user_sent_vec,tgt_agent_sent_vec,src_sent_vec,mask_tgt_lens[0],mask_tgt_lens[1])

        # user_dec_state = self.user_decoder.init_decoder_state(src, top_vec)
        # agent_dec_state = self.agent_decoder.init_decoder_state(src, top_vec)

        # print(src_tgt_mask_final[0].size())
        # print(tgts[0][:, :-1].size())
        # user_outputs, agent_outputs, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent = self.role_decoder(tgts[0][:, :-1], tgts[1][:, :-1], top_vec, user_dec_state, agent_dec_state, merge_type, inter_weight, role_mask, src_tgt_mask_final, tgt_seg_idxs, mask_tgt_seg_idxs)
        
        #TODO: 针对将user 与 agent进行统一操作：
        #TODO: 在tgt中区分 user与agent 的mask:
        # print(tgt_fin_role[:,:-1])
        # print(self.user_tag_id)
        tgt_user_mask = tgt_fin_role[:,:-1].eq(self.user_tag_id)
        tgt_agent_mask = tgt_fin_role[:,:-1].eq(self.agent_tag_id)
        
        # 解码调用：
        tgt_dec_state = self.tog_decoder.init_decoder_state(src, top_vec)
        # TODO: 加上real utt mask(生成subsummary所关注的utt)
        # outputs, state, user_scores, agent_scores, last_layer_score = self.tog_decoder(tgts[:, :-1],top_vec,tgt_dec_state,role_mask,tgt_user_mask,tgt_agent_mask,src_tgt_mask_final)
        outputs, state, last_layer_score = self.tog_decoder(tgts[:, :-1],top_vec,tgt_dec_state,role_mask,tgt_user_mask,tgt_agent_mask,src_tgt_mask_final)
        
        # 可视化:
        # print(last_layer_score[0].size())
        # print(tgt_fin_role)
        # print(last_layer_score[0].sum(-1))
        '''
        att_utt,_ = self.pooling(word_vectors=last_layer_score[0], sent_lengths=tgt_fin_len, sent_lengths_mask=tgt_fin_len.bool())
        # print(att_utt.sum(-1))
        att_utt = att_utt.permute(0,2,1)
        att_utt,_ = self.pooling1(word_vectors=att_utt, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens)
        att_utt = att_utt.permute(0,2,1)
        '''
        # print(att_utt.sum(-1))
        # att_utt = att_utt.permute(0,2,1)

        # print(list(att_utt.cpu().numpy()))
        
        
        batch, tgt_len, src_len = src.size(0) ,tgt_user_mask.size(1), src.size(1)

        tgt_user_mask = tgt_user_mask.unsqueeze(2).expand(batch, tgt_len, src_len)
        tgt_agent_mask = tgt_agent_mask.unsqueeze(2).expand(batch, tgt_len, src_len)

        #TODO: 针对context utt hidden state 进行pointer-net generator的编写：
        # 使用和不使用 context utt layer 对应的 p-n generator 不同：
        # 对于 together decoding 的 PGN:
        # 首先计算每一步的src 关注分布与 summary hidden output 拼接：b, tgt, h*3
        # print(last_layer_score[0].size())
        # output_cat = torch.cat((torch.bmm(last_layer_score[0],top_vec),torch.bmm(last_layer_score[1],top_vec), outputs),-1)
        output_cat = torch.cat((torch.bmm(last_layer_score[0],top_vec), outputs),-1)

        # return [outputs,output_cat], None, [user_scores, agent_scores, tgt_user_mask, tgt_agent_mask], last_layer_score
        return [outputs,output_cat], None, None, last_layer_score
        # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent], scores
        # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent]

    # 强化学习：
    def rllearn(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final):
        # first, 整理好src rep
        top_vec = self.bert(src, segs, mask_src)
        src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens)
        # 从Start seg开始进入inference part:
        # 每次都是得到tgt_ids 然后进行和src_vec的一个score计算：
        # 循环终止条件为最终字段为END_ID
        tgt_seq = torch.full([top_vec.size(0), 1], self.start_token).to(src)
        # 设置保存seg_idxs和sent_lens的列表：
        tgt_seg_ids, tgt_sent_lens = [[],[]], [[],[]]
        for k in range(2):
            tgt_seq_vec = self.user_tgt_layer()


    # 用于生成base summary文本的inference阶段
    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        #print(max_length)
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        role_mask = batch.role_mask

        # 直接传入：
        src_features = self.model.bert(src, segs, mask_src)

        device = src_features.device
        decoders = [self.model.user_decoder, self.model.agent_decoder]
        results_full = []

        # decode role summaries
        dec_states_roles, src_features_p_roles = [], []
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq_roles, topk_log_probs_roles, hypotheses_roles, results_roles = [], [], [], []
        end_states = []
        role_masks = []
        # 一些初始化
        for k in range(2):
            dec_states_roles.append(decoders[k].init_decoder_state(src, src_features, with_cache=True))
            # Tile states and memory beam_size times.
            # beam size等于一的话没有区别
            dec_states_roles[k].map_batch_fn(
                lambda state, dim: tile(state, beam_size, dim=dim))
            src_features_p_roles.append(tile(src_features, beam_size, dim=0))
            role_masks.append(tile(role_mask, beam_size, dim=0))
            # B,1 Start_id
            alive_seq = torch.full(
                [batch_size * beam_size, 1],
                self.start_token,
                dtype=torch.long,
                device=device)
            alive_seq_roles.append(alive_seq)

            # Give full probability to the first beam on the first step.
            topk_log_probs = (
                torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                             device=device).repeat(batch_size))
            topk_log_probs_roles.append(topk_log_probs)

            # Structure that holds finished hypotheses.
            hypotheses = [[] for _ in range(batch_size)]  # noqa: F812
            hypotheses_roles.append(hypotheses)

            results = {}
            results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["gold_score"] = [0] * batch_size
            results["batch"] = batch
            results_roles.append(results)
            # [B*False]
            end_states.append(torch.zeros(batch_size).bool().to(device))

        for step in range(max_length):
            # print(step)
            # print('--------------------')
            # print(end_states)
            # print(hypotheses_roles)
            # 取当前的last hidden 1,B
            decoder_input_user = alive_seq_roles[0][:, -1].view(1, -1)
            decoder_input_agent = alive_seq_roles[1][:, -1].view(1, -1)
            # Decoder forward. B,1
            decoder_input_user = decoder_input_user.transpose(0,1)
            decoder_input_agent = decoder_input_agent.transpose(0, 1)
            user_outputs, agent_outputs, state_user, state_agent, _, _, _, _ = self.model.role_decoder(decoder_input_user, decoder_input_agent,
                                                                                     src_features_p_roles[0], dec_states_roles[0],
                                                                                     dec_states_roles[1], self.args.merge,
                                                                                     self.args.inter_weight, role_masks[0], step=step)

            # Generator forward.
            outputs = [user_outputs, agent_outputs]
            states = [state_user, state_agent]
            # 每一个step都会更新
            is_finished_roles, topk_score_roles = [], []
            batch_indexes = []
            # inference更新之后的落实操作：
            for k in range(2):
                log_probs = self.generator.forward(outputs[k].transpose(0,1).squeeze(0))
                vocab_size = log_probs.size(-1)

                if step < min_length:
                    log_probs[:, self.end_token] = -1e20

                # Multiply probs by the beam probability.
                #print(topk_log_probs_roles[k].shape)
                log_probs += topk_log_probs_roles[k].view(-1).unsqueeze(1)  # [batch*beam, vocab]

                alpha = self.global_scorer.alpha
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                # Flatten probs into a list of possibilities.
                curr_scores = log_probs / length_penalty  # [batch*beam, vocab]

                if(self.args.block_trigram):
                    cur_len = alive_seq_roles[k].size(1)
                    if(cur_len>5):
                        for i in range(alive_seq_roles[k].size(0)):
                            fail = False
                            words = [int(w) for w in alive_seq_roles[k][i]]
                            words = [self.vocab.ids_to_tokens[w] for w in words]
                            words = ' '.join(words).replace(' ##','').split()
                            if(len(words)<=5):
                                continue
                            trigrams = [(words[i-2],words[i-1],words[i],words[i+1], words[i+2]) for i in range(2,len(words)-2)]
                            trigram = tuple(trigrams[-1])
                            if trigram in trigrams[:-1]:
                                fail = True
                            if fail:
                                curr_scores[i] = -10e20

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                # B, vocab_size -> B, 1
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                # Recover log probs.
                topk_log_probs = topk_scores * length_penalty
                topk_log_probs_roles[k] = topk_log_probs

                # Resolve beam origin and true word ids.
                # 相当于 / 除法
                topk_beam_index = topk_ids.true_divide(vocab_size)
                # 除数除以元素的余数
                topk_ids = topk_ids.fmod(vocab_size) # B, 1

                # Map beam_index to batch_index in the flat representation.
                batch_index = (
                        topk_beam_index
                        + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                batch_indexes.append(batch_index)
                select_indices = batch_index.view(-1)

                # Append last prediction. token_id
                # B, cur_len
                alive_seq_roles[k] = torch.cat(
                    [alive_seq_roles[k].to(torch.long).index_select(0, select_indices.to(torch.long)),
                     topk_ids.to(torch.long).view(-1, 1)], -1)

                # 判断是否到end token id 
                is_finished = topk_ids.eq(self.end_token)  # [batch, beam]
                if step + 1 == max_length:
                    is_finished.fill_(1)
                # End condition is top beam is finished. |： 或
                # 或运算，只要有true便会体现出来：[B,1] -> [B] True:finish 
                end_states[k] = is_finished[:, 0].eq(1) | end_states[k]
                # is_finished_roles ： 防止出现user和agent结束时间不同
                is_finished_roles.append(is_finished)
                topk_score_roles.append(topk_scores)

                # 判断是否进入到<q>也即此轮segment结束：
                


            # 找到所有的unfinish的sample（只有user和agent都要生成完毕才行）
            end_condition = end_states[0] & end_states[1]
            non_finished = end_condition.eq(0).nonzero().view(-1)

            for k in range(2):
                # Save finished hypotheses.
                # B,1,cur_len
                predictions = alive_seq_roles[k].view(-1, beam_size, alive_seq_roles[k].size(-1))
                # batch
                for i in range(is_finished_roles[k].size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished_roles[k][i].fill_(1)
                    # 分别统计user和agent有哪些finish, batch_idx
                    # beam size为1时，finished_hyp大小为 0或1个
                    finished_hyp = is_finished_roles[k][i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    # hypotheses_roles : 2 * B 个 list组成
                    for j in finished_hyp:
                        # 对于确定的k,b也即确定了这个sample
                        # 存起来已经finish的user或者agent part
                        hypotheses_roles[k][b].append((
                            topk_score_roles[k][i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        # 选择最终分数最合适的句子作为结束
                        best_hyp = sorted(
                            hypotheses_roles[k][b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results_roles[k]["scores"][b].append(score)
                        results_roles[k]["predictions"][b].append(pred)
                # Remove finished batches for the next step.
                topk_log_probs_roles[k] = topk_log_probs_roles[k].index_select(0, non_finished)
                alive_seq_roles[k] = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq_roles[k].size(-1))

            #print(non_finished)
            # If all sentences are translated, no need to go further.
            if len(non_finished) == 0:
                break
            # batch中有的先结束生成有的后结束，需要处理
            batch_offset = batch_offset.index_select(0, non_finished)
            # Reorder states.
            for k in range(2):
                batch_indexes[k] = batch_indexes[k].index_select(0, non_finished)
                end_states[k] = end_states[k].index_select(0, non_finished)
                #print(batch_indexes[k].to(torch.long))
                select_indices = batch_indexes[k].view(-1)
                src_features_p_roles[k] = src_features_p_roles[k].index_select(0, select_indices.to(torch.long))
                role_masks[k] = role_masks[k].index_select(0, select_indices.to(torch.long))
                dec_states_roles[k].map_batch_fn(
                    lambda state, dim: state.index_select(dim, select_indices.to(torch.long)))

        for k in range(2):
            #print(results_roles[k])
            results_full.append(results_roles[k])

        return results_full
        