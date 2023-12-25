# import copy
# import jieba
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel, BertConfig
# from torch.nn.init import xavier_uniform_
# from torch.nn.utils.rnn import pad_sequence

# from models.decoder import TransformerDecoder, RoleDecoder
# import models.decoder_t as decoder_topic
# import models.decoder_tag as decoder_tag
# from models.encoder import Classifier, ExtTransformerEncoder, TgtTransformerEncoder, ContextTransformerEncoder
# # from models.encoder import Classifier, ExtTransformerEncoder
# from models.optimizers import Optimizer
# from crf import CRFLayer
# from models.pooling import Pooling
# from models.neural import SummaryAttention, SummaryBothAttention

# def build_optim(args, model, checkpoint):
#     """ Build optimizer """

#     if checkpoint is not None:
#         optim = checkpoint['optim'][0]
#         saved_optimizer_state_dict = optim.optimizer.state_dict()
#         optim.optimizer.load_state_dict(saved_optimizer_state_dict)
#         if args.visible_gpus != '-1':
#             for state in optim.optimizer.state.values():
#                 for k, v in state.items():
#                     if torch.is_tensor(v):
#                         state[k] = v.cuda()

#         if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
#             raise RuntimeError(
#                 "Error: loaded Adam optimizer from existing model" +
#                 " but optimizer state is empty")

#     else:
#         optim = Optimizer(
#             args.optim, args.lr, args.max_grad_norm,
#             beta1=args.beta1, beta2=args.beta2,
#             decay_method='noam',
#             warmup_steps=args.warmup_steps)

#     optim.set_parameters(list(model.named_parameters()))


#     return optim

# def build_optim_bert(args, model, checkpoint, bert_op):
#     """ Build optimizer """

#     if checkpoint is not None:
#         # KL loss finetine 需要变换
#         optim = checkpoint['optims'][0]
#         # optim = checkpoint['optim']
#         saved_optimizer_state_dict = optim.optimizer.state_dict()
#         optim.optimizer.load_state_dict(saved_optimizer_state_dict)
#         if args.visible_gpus != '-1':
#             for state in optim.optimizer.state.values():
#                 for k, v in state.items():
#                     if torch.is_tensor(v):
#                         state[k] = v.cuda()

#         if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
#             raise RuntimeError(
#                 "Error: loaded Adam optimizer from existing model" +
#                 " but optimizer state is empty")

#     else:
#         optim = Optimizer(
#             args.optim, args.lr_bert, args.max_grad_norm,
#             beta1=args.beta1, beta2=args.beta2,
#             decay_method='noam',
#             warmup_steps=args.warmup_steps_bert)

#     # params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model') or n.startswith('user_tgt_layer')
#     #             or n.startswith('agent_tgt_layer')]
#     params = [(n, p) for n, p in list(model.named_parameters()) if n in bert_op]
#     optim.set_parameters(params)


#     return optim

# def build_optim_dec(args, model, checkpoint, bert_op):
#     """ Build optimizer """

#     # 修改一下判断条件
#     # if checkpoint is None:
#     if checkpoint is not None:
#         optim = checkpoint['optims'][1]
#         saved_optimizer_state_dict = optim.optimizer.state_dict()
#         optim.optimizer.load_state_dict(saved_optimizer_state_dict)
#         if args.visible_gpus != '-1':
#             for state in optim.optimizer.state.values():
#                 for k, v in state.items():
#                     if torch.is_tensor(v):
#                         state[k] = v.cuda()

#         if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
#             raise RuntimeError(
#                 "Error: loaded Adam optimizer from existing model" +
#                 " but optimizer state is empty")

#     else:
#         optim = Optimizer(
#             args.optim, args.lr_dec, args.max_grad_norm,
#             beta1=args.beta1, beta2=args.beta2,
#             decay_method='noam',
#             warmup_steps=args.warmup_steps_dec)

#     # params = [(n, p) for n, p in list(model.named_parameters()) if not (n.startswith('bert.model') and n.startswith('user_tgt_layer')
#     #             and n.startswith('agent_tgt_layer'))]
#     params = [(n, p) for n, p in list(model.named_parameters()) if n not in bert_op]
#     optim.set_parameters(params)


#     return optim

# # 映射到vocab中去（这块应该可以进行改进）
# def get_generator(vocab_size, dec_hidden_size, device):
#     gen_func = nn.LogSoftmax(dim=-1)
#     generator = nn.Sequential(
#         nn.Linear(dec_hidden_size, vocab_size),
#         gen_func
#     )
#     generator.to(device)

#     return generator

# class Bert(nn.Module):
#     def __init__(self, large, temp_dir, finetune=False):
#         super(Bert, self).__init__()
#         if(large):
#             self.model = BertModel.from_pretrained('../pretrained/bert_base_chinese', cache_dir=temp_dir)
#         else:
#             self.model = BertModel.from_pretrained('../pretrained/bert_base_chinese', cache_dir=temp_dir)

#         self.finetune = finetune

#     def forward(self, x, segs, mask):
#         if(self.finetune):
#             # top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
#             # 输出格式变了，只需要last_hidden_states
#             top_vec = self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state
#             # print(self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state.size())
#         else:
#             self.eval()
#             with torch.no_grad():
#                 top_vec = self.model(x, token_type_ids = segs, attention_mask=mask).last_hidden_state
#                 # top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
#         return top_vec


# class ExtSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint):
#         super(ExtSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

#         self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#                                                args.ext_dropout, args.ext_layers, args.label_class)
#         # self.tag_to_ix = {"B": 1, "I": 2, "O": 0, "<START>": 3, "<STOP>": 4}
#         # self.crf_layer = CRFLayer(self.tag_to_ix)

#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
#                                      num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
#             self.bert.model = BertModel(bert_config)
#             self.ext_layer = Classifier(self.bert.model.config.hidden_size)

#         if(args.max_pos>512):
#             my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#             my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#             my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#             self.bert.model.embeddings.position_embeddings = my_pos_embeddings

#         self.vocab_size = self.bert.model.config.vocab_size
#         tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#         if (self.args.share_emb):
#             tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#         # 在加一个专门用来训练decoder部分gold label的transformer encoder
#         self.user_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#                                                args.tgt_dropout, args.tgt_layers, tgt_embeddings)
#         self.agent_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#                                                args.tgt_dropout, args.tgt_layers, tgt_embeddings)
#         # get sent context info 
#         self.src_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#                                                args.ext_dropout, 1)
#         self.user_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#                                                args.tgt_dropout, 1)
#         self.agent_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#                                                args.tgt_dropout, 1)
#         # 得到句向量池化层：
#         self.pooling = Pooling(args.sent_rep_tokens, args.mean_tokens, args.max_tokens)
#         # 得分函数：
#         # self.user_score_layer = SummaryAttention(self.bert.model.config.hidden_size)
#         # self.agent_score_layer = SummaryAttention(self.bert.model.config.hidden_size)

#         self.score_layer = SummaryBothAttention(self.bert.model.config.hidden_size)

#         self.classifier = nn.Linear(self.bert.model.config.hidden_size*2, 2, bias=True)
#         self.party_embedding = nn.Embedding(3, self.bert.model.config.hidden_size)

#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             if args.param_init != 0.0:
#                 for p in self.ext_layer.parameters():
#                     p.data.uniform_(-args.param_init, args.param_init)
#             if args.param_init_glorot:
#                 for p in self.ext_layer.parameters():
#                     if p.dim() > 1:
#                         xavier_uniform_(p)
#             # initialize params.
#             if args.encoder == "transformer":
#                 for module in self.encoder.modules():
#                     self._set_parameter_tf(module)
#             for module in self.user_tgt_layer.modules():
#                 self._set_parameter_tf(module)
#             for module in self.agent_tgt_layer.modules():
#                 self._set_parameter_tf(module)
#             for module in self.src_context_layer.modules():
#                 self._set_parameter_tf(module)
#             for module in self.user_tgt_context_layer.modules():
#                 self._set_parameter_tf(module)
#             for module in self.agent_tgt_context_layer.modules():
#                 self._set_parameter_tf(module)
#             for p in self.score_layer.parameters():
#                 self._set_parameter_linear(p)
#             for p in self.classifier.parameters():
#                 self._set_parameter_linear(p)
#             for module in self.party_embedding.modules():
#                 self._set_parameter_tf(module)

#             if(args.use_bert_emb):
#                 tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#                 tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#                 self.user_tgt_layer.embeddings = tgt_embeddings
#                 self.agent_tgt_layer.embeddings = tgt_embeddings

#         self.to(device)

#     def _set_parameter_tf(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

#     def _set_parameter_linear(self, p):
#         if p.dim() > 1:
#             xavier_uniform_(p)
#         else:
#             p.data.zero_()

#     # def forward(self, src, segs, clss, mask_src, mask_cls):
#     def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final):
#         # 以utterance为单位进行学习看一下效果
#         # batch_size = src.shape[0]
#         # dialog_length = src.shape[1]
#         # utterance_length = src.shape[2]

#         # input_ids = src.view(batch_size * dialog_length, utterance_length)
#         # token_type_ids = segs.view(batch_size * dialog_length, utterance_length)
#         # attention_mask = mask_src.view(batch_size * dialog_length, utterance_length)
        
#         top_vec = self.bert(src, segs, mask_src)
#         # b*u_n, u_l, hidden
#         # top_vec = self.bert(input_ids, token_type_ids, attention_mask)
#         # 取cls -> b, u_n, hidden 
#         # sents_vec = top_vec.view(batch_size, dialog_length, utterance_length, -1)[:, :, 0]
#         # print(top_vec)
#         # [B,1]
#         # [B,cls_num,H] 
#         # clss ： [B,cls_num]
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         # src_sent_vec = sents_vec * mask_cls[:, :, None].float()
#         # # 对cls hidden states进行classify
#         # b,src,label
#         # role_mask : b,src
#         # party_embeddings = self.party_embedding(role_mask)
#         # src_sent_vec = torch.cat((sents_vec, party_embeddings), dim=-1)
#         # sent_scores = self.ext_layer(sents_vec, mask_cls, role_mask).squeeze(-1)
#         sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
#         # 直接先粗俗的取词向量平均(取max)转化为句向量：

#         ########################################################################
#         # src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
#         # 对golden label 进行编码：
#         # tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
#         # tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
#         # # sentence context 计算：
#         # src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
#         # tgt_user_sent_vec = self.user_tgt_context_layer(tgt_user_sent_vec,mask_tgt_lens[0],True)
#         # tgt_agent_sent_vec = self.agent_tgt_context_layer(tgt_agent_sent_vec,mask_tgt_lens[1],True)
#         # Score函数(user and agent) batch, target_l, source_l, 2
#         # 此处考虑agent和user的合二为一看看怎么样
#         #########################################################################

#         # user_utt_scores = self.user_score_layer(tgt_user_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[0])
#         # agent_utt_scores = self.user_score_layer(tgt_agent_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[1])

#         # batch, target_l, source_l, 2
#         # scores = self.score_layer(tgt_user_sent_vec,tgt_agent_sent_vec,src_sent_vec,mask_tgt_lens[0],mask_tgt_lens[1])
#         #######直接对utterance分类我看一下怎么样效果：(句子分类对应ext任务)
#         # B, utterance_num, 2 
#         # sent_scores = self.classifier(src_sent_vec)
        
#         # return [user_utt_scores,agent_utt_scores]
#         return sent_scores
#         # CRF
#         #crf_scores = self.crf_layer.neg_log_likelihood(sent_scores, tgt)
#         #return crf_scores, mask_cls

#     def decode(self, src, segs, clss, mask_src, mask_cls, tgt):
#         top_vec = self.bert(src, segs, mask_src)
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         sents_vec = sents_vec * mask_cls[:, :, None].float()
#         sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
#         tag_seq = self.crf_layer.forward_test(sent_scores)
#         return tag_seq


# class AbsSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint=None, bert_from_extractive=None, tokenizer=None, label2vid=None):
#         super(AbsSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

#         if bert_from_extractive is not None:
#             # 打印出来看一看：
#             # for n, p in bert_from_extractive.items():
#             #     print(n)
            
#             self.bert.model.load_state_dict(
#                 dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                      num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                      intermediate_size=args.enc_ff_size,
#                                      hidden_dropout_prob=args.enc_dropout,
#                                      attention_probs_dropout_prob=args.enc_dropout)
#             self.bert.model = BertModel(bert_config)

#         if(args.max_pos>512):
#             my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#             my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#             my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#             self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#         # embedding set 
#         # 要对bert的embedding部分进行一个resize
#         num_tokens, _ = self.bert.model.embeddings.word_embeddings.weight.shape
#         self.bert.model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)

#         # 此处来设置decoder embdding part对应encoder的embedding （即share embedding）
#         self.vocab_size = self.bert.model.config.vocab_size 
#         # tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#         topic_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)

#         if (self.args.share_emb):
#             # tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#             assert topic_embeddings.weight.shape[0] == self.bert.model.embeddings.word_embeddings.weight.shape[0]
#             topic_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

#         # 增加一个decode topic sequence的部分：
#         self.topic_decoder = decoder_topic.TransformerDecoder(
#             self.args.dec_layers,
#             self.args.dec_hidden_size, heads=self.args.dec_heads,
#             d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=topic_embeddings)
        
#         self.tag_decoder = decoder_tag.TransformerDecoder(
#             self.args.dec_layers,
#             self.args.dec_hidden_size, heads=self.args.dec_heads,
#             d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=topic_embeddings)

#         # 增加一个utterence级别的context encoder:
#         self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#                                                args.ext_dropout, args.ext_layers)
#         # 一些索引：
#         self.label_ids = list(label2vid.values())
#         # self.label_start_id = min(label_ids)
#         # self.label_end_id = max(label_ids)+1
#         self.pad_token_id = 0
#         # 后续映射tag标签使用
#         # self.mapping = torch.LongTensor([99, 1]+self.label_ids)
#         self.mapping = torch.LongTensor([99, 1, 2])
#         self.src_start_index = len(self.mapping)
#         self.tag_index = 2
#         # torch.empty: 用来返回一个没有初始化的tensor
#         self.pn_init_token = nn.Parameter(torch.empty([1, self.bert.model.config.hidden_size]))
#         self.terminate_state = nn.Parameter(torch.empty([1, self.bert.model.config.hidden_size]))

#         # 对topic decoder 的embedding 改动
#         for token in tokenizer.unique_no_split_tokens:
#             if token[:2] == '<<':  # 特殊字符
#                 # tag embedding 由 origin token embdding 得到
#                 # 从label2vid mapping中对应提取应该是最保险的：
#                 index = label2vid[token]
#                 # index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
#                 # if len(index)>1:
#                 #     raise RuntimeError(f"{token} wrong split")
#                 # else:
#                 #     # 取出其在vocab中对应的index
#                 #     index = index[0]
#                 assert index>=num_tokens, (index, num_tokens, token)
#                 # topic token embedding mean 
#                 # 使用jieba对topic进行分词
#                 tag_token = ' '.join(list(jieba.cut(token[2:-2])))
#                 # 都是已有的词汇id
#                 indexes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tag_token))
#                 # 确认一下
#                 for idx in indexes:
#                     assert idx < num_tokens

#                 embed = self.bert.model.embeddings.word_embeddings.weight.data[indexes[0]]
#                 for i in indexes[1:]:
#                     embed += self.bert.model.embeddings.word_embeddings.weight.data[i]
#                 embed /= len(indexes)
#                 self.topic_decoder.embeddings.weight.data[index] = embed
        
#         # mlp layer like BARTNER:
#         hidden_size = self.bert.model.config.hidden_size
#         self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
#                                             nn.Dropout(0.3),
#                                             nn.ReLU(),
#                                             nn.Linear(hidden_size, hidden_size))
#         self.dropout_layer = nn.Dropout(0.3)
#         # 在加一个专门用来训练decoder部分gold label的transformer encoder
#         # self.user_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#         #                                        args.tgt_dropout, args.tgt_layers, tgt_embeddings)
#         # self.agent_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#         #                                        args.tgt_dropout, args.tgt_layers, tgt_embeddings)
#         # # get sent context info 
#         # self.src_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#         #                                        args.ext_dropout, 1)
#         # self.user_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#         #                                        args.tgt_dropout, 1)
#         # self.agent_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
#         #                                        args.tgt_dropout, 1)
#         # # 得到句向量池化层：
#         # self.pooling = Pooling(args.sent_rep_tokens, args.mean_tokens, args.max_tokens)
#         # # 得分函数：
#         # self.user_score_layer = SummaryAttention(self.bert.model.config.hidden_size)
#         # self.agent_score_layer = SummaryAttention(self.bert.model.config.hidden_size)

#         # self.user_decoder = TransformerDecoder(
#         #     self.args.dec_layers,
#         #     self.args.dec_hidden_size, heads=self.args.dec_heads,
#         #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
#         # self.agent_decoder = TransformerDecoder(
#         #     self.args.dec_layers,
#         #     self.args.dec_hidden_size, heads=self.args.dec_heads,
#         #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
#         # self.role_decoder = RoleDecoder(self.user_decoder, self.agent_decoder)
#         # todo: different generator? make it an option, first set it to be the same
#         # self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
#         # self.generator[0].weight = self.user_decoder.embeddings.weight

#         # 这块不太对，应该来说是要初始化decoder部分，但是之前的ext只有encoder部分
#         # train 和 test 要来回改这块，很麻烦
#         # if checkpoint is None:
#         #     pass
#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             if args.param_init != 0.0:
#                 for p in self.ext_layer.parameters():
#                     p.data.uniform_(-args.param_init, args.param_init)
#             if args.param_init_glorot:
#                 for p in self.ext_layer.parameters():
#                     if p.dim() > 1:
#                         xavier_uniform_(p)
#             # for module in self.user_decoder.modules():
#             #     if isinstance(module, (nn.Linear, nn.Embedding)):
#             #         module.weight.data.normal_(mean=0.0, std=0.02)
#             #     elif isinstance(module, nn.LayerNorm):
#             #         module.bias.data.zero_()
#             #         module.weight.data.fill_(1.0)
#             #     if isinstance(module, nn.Linear) and module.bias is not None:
#             #         module.bias.data.zero_()
#             # for module in self.agent_decoder.modules():
#             #     if isinstance(module, (nn.Linear, nn.Embedding)):
#             #         module.weight.data.normal_(mean=0.0, std=0.02)
#             #     elif isinstance(module, nn.LayerNorm):
#             #         module.bias.data.zero_()
#             #         module.weight.data.fill_(1.0)
#             #     if isinstance(module, nn.Linear) and module.bias is not None:
#             #         module.bias.data.zero_()
#             for module in self.topic_decoder.modules():
#                 # if isinstance(module, (nn.Linear, nn.Embedding)):
#                 # 由于embedding使用bert的故不再此进行初始化
#                 if isinstance(module, nn.Linear):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                 elif isinstance(module, nn.LayerNorm):
#                     module.bias.data.zero_()
#                     module.weight.data.fill_(1.0)
#                 if isinstance(module, nn.Linear) and module.bias is not None:
#                     module.bias.data.zero_()
#             for module in self.tag_decoder.modules():
#                 # if isinstance(module, (nn.Linear, nn.Embedding)):
#                 # 由于embedding使用bert的故不再此进行初始化
#                 if isinstance(module, nn.Linear):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                 elif isinstance(module, nn.LayerNorm):
#                     module.bias.data.zero_()
#                     module.weight.data.fill_(1.0)
#                 if isinstance(module, nn.Linear) and module.bias is not None:
#                     module.bias.data.zero_()
#             # for p in self.generator.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             for p in self.encoder_mlp.parameters():
#                 self._set_parameter_linear(p)
#             self._set_parameter_linear(self.pn_init_token)
#             self._set_parameter_linear(self.terminate_state)
#             # for p in self.src_context_layer.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             # for p in self.user_tgt_context_layer.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             # for p in self.agent_tgt_context_layer.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             # for p in self.user_score_layer.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             # for p in self.agent_score_layer.parameters():
#             #     if p.dim() > 1:
#             #         xavier_uniform_(p)
#             #     else:
#             #         p.data.zero_()
#             if(args.use_bert_emb):
#                 pass
#                 # tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#                 # tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#                 # self.user_decoder.embeddings = tgt_embeddings
#                 # self.agent_decoder.embeddings = tgt_embeddings
#                 # self.user_tgt_layer.embeddings = tgt_embeddings
#                 # self.agent_tgt_layer.embeddings = tgt_embeddings
#                 # self.generator[0].weight = self.user_decoder.embeddings.weight

#         self.to(device)

#     def _set_parameter_linear(self, p):
#         if p.dim() > 1:
#             xavier_uniform_(p)
#         else:
#             p.data.zero_()

#     # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final):
#     # # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, src_tgt_mask_final):
#     #     top_vec = self.bert(src, segs, mask_src)
#     #     #### 分别得到src与tgt的encoder vector 
#     #     # 直接先粗俗的取词向量平均(取max)转化为句向量：
#     #     src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
#     #     # 对golden label 进行编码：
#     #     tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
#     #     tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
#     #     # sentence context 计算：
#     #     src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
#     #     tgt_user_sent_vec = self.user_tgt_context_layer(tgt_user_sent_vec,mask_tgt_lens[0],True)
#     #     tgt_agent_sent_vec = self.agent_tgt_context_layer(tgt_agent_sent_vec,mask_tgt_lens[1],True)
#     #     # Score函数(user and agent) batch, target_l, source_l, 2
#     #     user_utt_scores = self.user_score_layer(tgt_user_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[0])
#     #     agent_utt_scores = self.user_score_layer(tgt_agent_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[1])

#     #     user_dec_state = self.user_decoder.init_decoder_state(src, top_vec)
#     #     agent_dec_state = self.agent_decoder.init_decoder_state(src, top_vec)

#     #     # print(src_tgt_mask_final[0].size())
#     #     # print(tgts[0][:, :-1].size())
#     #     user_outputs, agent_outputs, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent = self.role_decoder(tgts[0][:, :-1], tgts[1][:, :-1], top_vec, user_dec_state, agent_dec_state, merge_type, inter_weight, role_mask, src_tgt_mask_final)
        
#     #     return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent],[user_utt_scores, agent_utt_scores]
#     #     # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent]
#     # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, src_tgt_mask_final):
    
#     def forward(self, src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls, avg_feature=True):
#         top_vec = self.bert(src, segs, mask_src)
#         # 添加一个context encoder 将src的 token 进化到 utterence 层面上：
#         # TODO: 使用cls作为utterence的表示或者使用mean/max等表示（待测试）
#         # b, src_utt_len, h
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         context_vec = self.ext_layer(sents_vec, mask_cls)
#         # 使用类似BARTNER的方式进行decoder：
#         dec_state = self.topic_decoder.init_decoder_state(sents_vec, context_vec)
#         # decoder part:
#         # 对tgt输入topic_tgt 进行修改；
#         # topic_tgt : b, topic_sequence
#         # 找哪些位置是tag token 哪些位置是src token 
#         mapping_token_mask = topic_tgt.lt(self.src_start_index)  
#         # ge: 大于等于 （将src部分mask掉）
#         mapped_tokens = topic_tgt.masked_fill(topic_tgt.ge(self.src_start_index), 0)
#         # 映射到在vocab中的位置： 99，99，99... tag_index,...,1
#         # print(mapped_tokens)
#         tag_mapped_tokens = self.mapping[mapped_tokens].to(mapped_tokens)
#         tag_mapped_tokens = tag_mapped_tokens.masked_fill(~mask_topic_target, self.pad_token_id)
#         # print(tag_mapped_tokens)
#         # 找到src token的位置：
#         src_tokens_index = topic_tgt - self.src_start_index # bsz x tgt_num
#         src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
#         # print(src_tokens_index)
#         # 从相应位置中提取utterence向量作为对应的decoder输入：
#         for i in range(src_tokens_index.size(0)):
#             assert int(src_tokens_index[i].max()) < len(sents_vec[i])
#         utt_mapped_list = []
#         for i in range(sents_vec.size(0)):
#             utt_mapped_list.append(torch.index_select(sents_vec[i], 0, src_tokens_index[i]))
#         # b, tgt_len, h
#         utt_mapped_vec = torch.stack(utt_mapped_list).to(sents_vec)
#         # tag embedding : b, tgt_len, emb_h
#         tag_mapped_vec = self.topic_decoder.embeddings(tag_mapped_tokens)
#         assert tag_mapped_vec.dim() == 3  # len x batch x embedding_dim 
#         assert tag_mapped_vec.size(-1) == utt_mapped_vec.size(-1)
#         # 将 utt tensor 与 tag tensor 拼接起来作为decoder的输入：
#         # True : tag_mapped_tokens ; False : word_mapped_tokens
#         decoder_input_list = []
#         for i in range(mapping_token_mask.size(0)):
#             condition = mapping_token_mask[i].unsqueeze(-1)
#             # print(condition)
#             decoder_input_list.append(torch.where(condition, tag_mapped_vec[i], utt_mapped_vec[i]))
#         decoder_input = torch.stack(decoder_input_list).to(utt_mapped_vec)
#         assert decoder_input.dim() == 3
#         # 去掉最后一个eos token 然后输入decoder中：
#         topic_output, state_topic = self.topic_decoder(decoder_input[:,:-1,:],context_vec,dec_state,memory_masks=~mask_cls,tgt_masks=mask_topic_target[:,:-1])
#         # 对生成的 topic_output 进行处理得到并为后续计算loss
#         # b, tgt, vocab_size(tag+2+cur_utt_num)
#         logits = topic_output.new_full((topic_output.size(0), topic_output.size(1), self.src_start_index+sents_vec.size(1)),
#                                        fill_value=-1e24)

#         # 首先计算的是 end 标签以及 Tag 标签：
#         eos_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[1:2]))  # bsz x max_len x 1
#         pad_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[0:1]))
#         # mapping_ids = torch.LongTensor(self.label_ids)
#         # tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
#         # tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[mapping_ids]))
#         tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[2:3]))

#         # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
#         # b, utt_num, h
#         if hasattr(self, 'encoder_mlp'):
#             context_vec = self.encoder_mlp(context_vec)

#         # 先把feature合并一下 
#         sents_vec = self.dropout_layer(sents_vec)
#         if avg_feature: 
#             context_vec = (context_vec + sents_vec)/2
#         # 计算utt hidden 与 decoder hidden 相关得分
#         # b, tgt_len, utt_len 
#         word_scores = torch.einsum('blh,bnh->bln', topic_output, context_vec)  # bsz x max_len x max_word_len
#         if not avg_feature:
#             gen_scores = torch.einsum('blh,bnh->bln', topic_output, sents_vec)  # bsz x max_len x max_word_len
#             word_scores = (gen_scores + word_scores)/2
        
#         # 针对最后一维即utt padding 进行mask
#         mask = ~mask_cls.unsqueeze(1)
#         word_scores = word_scores.masked_fill(mask, -1e32)

#         # 最后整合到一起：
#         # sos 位置空缺保证输出结果没有sos标签：
#         logits[:, :, 0:1] = pad_scores
#         logits[:, :, 1:2] = eos_scores
#         logits[:, :, 2:self.src_start_index] = tag_scores
#         logits[:, :, self.src_start_index:] = word_scores

#         #TODO: 加上后面的decoder summary tgt token 模块（real utt mask 以及 提取出tgt tag部分作为监督信号）
#         # 对tag hidden state 进行打乱重排：
#         # Training part:

#         device = topic_output.device
#         dim = topic_output.size(-1)
#         tag_mask = topic_tgt.eq(self.tag_index)[:,:-1]
#         # tag tgt index(containing eos)
#         tag_num = tag_mask.sum(1)
#         tag_tgt_list = []
#         tag_src_mask_list = []
#         reshuffle_list = []
#         for i in range(tag_num.size(0)):
#             shuffle_index = [i for i in range(int(tag_num[i]))]
#             np.random.shuffle(shuffle_index)
#             # shuffle and reshuffle 
#             tag_tgt_list.append(torch.cat([torch.arange(1,int(tag_num[i])+1)[shuffle_index],torch.tensor([0])]))
#             reshuffle_list.append(np.argsort(shuffle_index))
#             tag_src_mask_list.append(torch.ones(int(tag_num[i])))
#         # b, max_tgt_num+1
#         tag_tgt = pad_sequence(tag_tgt_list,batch_first=True,padding_value=-1).to(device)
#         tag_src_mask = pad_sequence(tag_src_mask_list,batch_first=True,padding_value=0).bool().to(device)
#         tag_tgt_mask = ~(tag_tgt == -1)
#         tag_tgt[tag_tgt == -1] = 0
#         # tag_tgt_mask = pad_sequence(tag_tgt_mask_list,batch_first=True,padding_value=0)
#         # 选取相对应的hidden state
#         #TODO:根据index进行shuffle
#         tag_list = []
#         shuffle_tag_list = []
#         for i in range(topic_output.size(0)):
#             tag_list.append(torch.masked_select(topic_output[i],tag_mask[i].unsqueeze(-1)).reshape(-1,dim))
#             shuffle_tag_list.append(torch.masked_select(topic_output[i],tag_mask[i].unsqueeze(-1)).reshape(-1,dim)[reshuffle_list[i]])
#         # b, max_tag_num, h
#         # print(tag_list)
#         tag_vec = pad_sequence(tag_list,batch_first=True,padding_value=0).to(device)
#         re_tag_vec = pad_sequence(shuffle_tag_list,batch_first=True,padding_value=0).to(device)
#         # print(tag_vec.size())
#         tg_decoder_input = torch.cat([self.pn_init_token.unsqueeze(0).expand(topic_output.size(0), 1, -1), tag_vec], 1).to(device)
#         # tag reranker:
#         tag_state = self.tag_decoder.init_decoder_state(tag_vec, tag_vec)
#         assert tg_decoder_input.size(1) == tag_tgt_mask.size(1)
#         tag_output, _ = self.tag_decoder(tg_decoder_input,re_tag_vec,tag_state,memory_masks=~tag_src_mask,tgt_masks=~tag_tgt_mask)
#         # 计算tgt logits:
#         tag_src_vec = torch.cat([self.terminate_state.unsqueeze(0).expand(topic_output.size(0), 1, -1), re_tag_vec], 1)
#         # b, tag_tgt+sos, tag_tgt+eos
#         tag_rank_scores = torch.einsum('blh,bnh->bln', tag_output, tag_src_vec)
#         mask = ~tag_tgt_mask.unsqueeze(1)
#         rank_logits = tag_rank_scores.masked_fill(mask, -1e32)

#         return logits, rank_logits, tag_tgt, tag_tgt_mask
#         # return logits
#         # user_dec_state = self.user_decoder.init_decoder_state(src, top_vec)
#         # agent_dec_state = self.agent_decoder.init_decoder_state(src, top_vec)
#         # print(tgts[0][:, :-1])
#         # user_outputs, agent_outputs, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent = self.role_decoder(tgts[0][:, :-1], tgts[1][:, :-1], top_vec, user_dec_state, agent_dec_state, merge_type, inter_weight, role_mask, src_tgt_mask_final)
#         # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent]

import copy
import jieba
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from models.decoder import TransformerDecoder, RoleDecoder
import models.decoder_t as decoder_topic
import models.decoder_tag as decoder_tag
# import models.decoder_all as decoder_all
from models.encoder import Classifier, ExtTransformerEncoder, TgtTransformerEncoder, ContextTransformerEncoder
# from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from crf import CRFLayer
from models.pooling import Pooling
from models.neural import SummaryAttention, SummaryBothAttention

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

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers, args.label_class)
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

        self.classifier = nn.Linear(self.bert.model.config.hidden_size*2, 2, bias=True)
        self.party_embedding = nn.Embedding(3, self.bert.model.config.hidden_size)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
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
            for module in self.party_embedding.modules():
                self._set_parameter_tf(module)

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
        # 以utterance为单位进行学习看一下效果
        # batch_size = src.shape[0]
        # dialog_length = src.shape[1]
        # utterance_length = src.shape[2]

        # input_ids = src.view(batch_size * dialog_length, utterance_length)
        # token_type_ids = segs.view(batch_size * dialog_length, utterance_length)
        # attention_mask = mask_src.view(batch_size * dialog_length, utterance_length)
        
        top_vec = self.bert(src, segs, mask_src)
        # b*u_n, u_l, hidden
        # top_vec = self.bert(input_ids, token_type_ids, attention_mask)
        # 取cls -> b, u_n, hidden 
        # sents_vec = top_vec.view(batch_size, dialog_length, utterance_length, -1)[:, :, 0]
        # print(top_vec)
        # [B,1]
        # [B,cls_num,H] 
        # clss ： [B,cls_num]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # src_sent_vec = sents_vec * mask_cls[:, :, None].float()
        # # 对cls hidden states进行classify
        # b,src,label
        # role_mask : b,src
        # party_embeddings = self.party_embedding(role_mask)
        # src_sent_vec = torch.cat((sents_vec, party_embeddings), dim=-1)
        # sent_scores = self.ext_layer(sents_vec, mask_cls, role_mask).squeeze(-1)
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        # 直接先粗俗的取词向量平均(取max)转化为句向量：

        ########################################################################
        # src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
        # 对golden label 进行编码：
        # tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
        # tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
        # # sentence context 计算：
        # src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
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
        # sent_scores = self.classifier(src_sent_vec)
        
        # return [user_utt_scores,agent_utt_scores]
        return sent_scores
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
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None, tokenizer=None, label2vid=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
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
        # 要对bert的embedding部分进行一个resize
        num_tokens, _ = self.bert.model.embeddings.word_embeddings.weight.shape
        self.bert.model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)

        # 此处来设置decoder embdding part对应encoder的embedding （即share embedding）
        self.vocab_size = self.bert.model.config.vocab_size 
        self.tgt_vocab_size = num_tokens

        tgt_embeddings = nn.Embedding(num_tokens, self.bert.model.config.hidden_size, padding_idx=0)
        topic_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)

        if (self.args.share_emb):
            # tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
            assert topic_embeddings.weight.shape[0] == self.bert.model.embeddings.word_embeddings.weight.shape[0]
            topic_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
            # 舍弃special token部分：
            tgt_embeddings.weight.data = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight.data)[:num_tokens]

        # 增加一个decode topic sequence的部分：
        self.topic_decoder = decoder_topic.TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=topic_embeddings)
        
        self.tag_decoder = decoder_tag.TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=topic_embeddings)

        # TODO:添加上最终summary解码部分：
        # 一体化解码：
        # self.tog_decoder = decoder_all.TransformerDecoder(
        #     self.args.dec_layers,
        #     self.args.dec_hidden_size, heads=self.args.dec_heads,
        #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        # todo: different generator? make it an option, first set it to be the same
        # self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator = CopyGenerator(num_tokens, self.args.dec_hidden_size, device)

        # 增加一个utterence级别的context encoder:
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        # 一些索引：
        self.label_ids = list(label2vid.values())
        # self.label_start_id = min(label_ids)
        # self.label_end_id = max(label_ids)+1
        self.pad_token_id = 0
        # 后续映射tag标签使用
        # self.mapping = torch.LongTensor([99, 1]+self.label_ids)
        self.mapping = torch.LongTensor([99, 1, 2])
        self.src_start_index = len(self.mapping)
        self.tag_index = 2
        # torch.empty: 用来返回一个没有初始化的tensor
        self.pn_init_token = nn.Parameter(torch.empty([1, self.bert.model.config.hidden_size]))
        self.terminate_state = nn.Parameter(torch.empty([1, self.bert.model.config.hidden_size]))

        self.user_tag_id, self.agent_tag_id = 1,2

        # 对topic decoder 的embedding 改动
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                # tag embedding 由 origin token embdding 得到
                # 从label2vid mapping中对应提取应该是最保险的：
                index = label2vid[token]
                # index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                # if len(index)>1:
                #     raise RuntimeError(f"{token} wrong split")
                # else:
                #     # 取出其在vocab中对应的index
                #     index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                # topic token embedding mean 
                # 使用jieba对topic进行分词
                tag_token = ' '.join(list(jieba.cut(token[2:-2])))
                # 都是已有的词汇id
                indexes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tag_token))
                # 确认一下
                for idx in indexes:
                    assert idx < num_tokens

                embed = self.bert.model.embeddings.word_embeddings.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += self.bert.model.embeddings.word_embeddings.weight.data[i]
                embed /= len(indexes)
                self.topic_decoder.embeddings.weight.data[index] = embed
        
        # mlp layer like BARTNER:
        hidden_size = self.bert.model.config.hidden_size
        self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                            nn.Dropout(0.3),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, hidden_size))
        self.dropout_layer = nn.Dropout(0.3)
        # 在加一个专门用来训练decoder部分gold label的transformer encoder
        # self.user_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
        #                                        args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        # self.agent_tgt_layer = TgtTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
        #                                        args.tgt_dropout, args.tgt_layers, tgt_embeddings)
        # # get sent context info 
        # self.src_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
        #                                        args.ext_dropout, 1)
        # self.user_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
        #                                        args.tgt_dropout, 1)
        # self.agent_tgt_context_layer = ContextTransformerEncoder(self.bert.model.config.hidden_size, args.tgt_ff_size, args.tgt_heads,
        #                                        args.tgt_dropout, 1)
        # # 得到句向量池化层：
        # self.pooling = Pooling(args.sent_rep_tokens, args.mean_tokens, args.max_tokens)
        # # 得分函数：
        # self.user_score_layer = SummaryAttention(self.bert.model.config.hidden_size)
        # self.agent_score_layer = SummaryAttention(self.bert.model.config.hidden_size)

        # self.user_decoder = TransformerDecoder(
        #     self.args.dec_layers,
        #     self.args.dec_hidden_size, heads=self.args.dec_heads,
        #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        # self.agent_decoder = TransformerDecoder(
        #     self.args.dec_layers,
        #     self.args.dec_hidden_size, heads=self.args.dec_heads,
        #     d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        # self.role_decoder = RoleDecoder(self.user_decoder, self.agent_decoder)
        # todo: different generator? make it an option, first set it to be the same
        # self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        # self.generator[0].weight = self.user_decoder.embeddings.weight

        # 这块不太对，应该来说是要初始化decoder部分，但是之前的ext只有encoder部分
        # train 和 test 要来回改这块，很麻烦
        # if checkpoint is None:
        #     pass
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)
            # for module in self.tog_decoder.modules():
            #     # if isinstance(module, (nn.Linear, nn.Embedding)):
            #     # 由于embedding使用bert的故不再此进行初始化
            #     if isinstance(module, nn.Linear):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()
            # for module in self.generator.modules():
            #     if isinstance(module, nn.Linear):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()
            # if(args.use_bert_emb):
            #     self.generator.out2.weight = self.tog_decoder.embeddings.weight
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
            # for module in self.user_decoder.modules():
            #     if isinstance(module, (nn.Linear, nn.Embedding)):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()
            # for module in self.agent_decoder.modules():
            #     if isinstance(module, (nn.Linear, nn.Embedding)):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()
            for module in self.topic_decoder.modules():
                # if isinstance(module, (nn.Linear, nn.Embedding)):
                # 由于embedding使用bert的故不再此进行初始化
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.tag_decoder.modules():
                # if isinstance(module, (nn.Linear, nn.Embedding)):
                # 由于embedding使用bert的故不再此进行初始化
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            # for module in self.tog_decoder.modules():
            #     # if isinstance(module, (nn.Linear, nn.Embedding)):
            #     # 由于embedding使用bert的故不再此进行初始化
            #     if isinstance(module, nn.Linear):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()
            for module in self.generator.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            # for p in self.generator.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            for p in self.encoder_mlp.parameters():
                self._set_parameter_linear(p)
            self._set_parameter_linear(self.pn_init_token)
            self._set_parameter_linear(self.terminate_state)
            # for p in self.src_context_layer.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            # for p in self.user_tgt_context_layer.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            # for p in self.agent_tgt_context_layer.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            # for p in self.user_score_layer.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            # for p in self.agent_score_layer.parameters():
            #     if p.dim() > 1:
            #         xavier_uniform_(p)
            #     else:
            #         p.data.zero_()
            if(args.use_bert_emb):
                pass
                # self.generator.out2.weight = self.tog_decoder.embeddings.weight
                # tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                # tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                # self.user_decoder.embeddings = tgt_embeddings
                # self.agent_decoder.embeddings = tgt_embeddings
                # self.user_tgt_layer.embeddings = tgt_embeddings
                # self.agent_tgt_layer.embeddings = tgt_embeddings
                # self.generator[0].weight = self.user_decoder.embeddings.weight

        self.to(device)

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens, mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final):
    # # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, src_tgt_mask_final):
    #     top_vec = self.bert(src, segs, mask_src)
    #     #### 分别得到src与tgt的encoder vector 
    #     # 直接先粗俗的取词向量平均(取max)转化为句向量：
    #     src_sent_vec,_ = self.pooling(word_vectors=top_vec, sent_lengths=src_lens, sent_lengths_mask=mask_src_lens) # B, S_L, H -> B,s_num,H
    #     # 对golden label 进行编码：
    #     tgt_user_sent_vec,_ = self.pooling(word_vectors=self.user_tgt_layer(tgts[0],tgt_seg_idxs[0],mask_tgt_seg_idxs[0],mask_tgts[0]), sent_lengths=tgt_lens[0], sent_lengths_mask=mask_tgt_lens[0])
    #     tgt_agent_sent_vec,_ = self.pooling(word_vectors=self.agent_tgt_layer(tgts[1],tgt_seg_idxs[1],mask_tgt_seg_idxs[1],mask_tgts[1]), sent_lengths=tgt_lens[1], sent_lengths_mask=mask_tgt_lens[1])
    #     # sentence context 计算：
    #     src_sent_vec = self.src_context_layer(src_sent_vec,mask_cls)
    #     tgt_user_sent_vec = self.user_tgt_context_layer(tgt_user_sent_vec,mask_tgt_lens[0],True)
    #     tgt_agent_sent_vec = self.agent_tgt_context_layer(tgt_agent_sent_vec,mask_tgt_lens[1],True)
    #     # Score函数(user and agent) batch, target_l, source_l, 2
    #     user_utt_scores = self.user_score_layer(tgt_user_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[0])
    #     agent_utt_scores = self.user_score_layer(tgt_agent_sent_vec,src_sent_vec,mask_cls,mask_tgt_lens[1])

    #     user_dec_state = self.user_decoder.init_decoder_state(src, top_vec)
    #     agent_dec_state = self.agent_decoder.init_decoder_state(src, top_vec)

    #     # print(src_tgt_mask_final[0].size())
    #     # print(tgts[0][:, :-1].size())
    #     user_outputs, agent_outputs, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent = self.role_decoder(tgts[0][:, :-1], tgts[1][:, :-1], top_vec, user_dec_state, agent_dec_state, merge_type, inter_weight, role_mask, src_tgt_mask_final)
        
    #     return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent],[user_utt_scores, agent_utt_scores]
    #     # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent]
    # def forward(self, src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, merge_type, inter_weight, role_mask, src_tgt_mask_final):
    
    # def forward(self, src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls, tgts, role_mask, tgt_fin_role, src_tgt_mask_final, avg_feature=True):
    def forward(self, src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls, avg_feature=True):
    # def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        # 添加一个context encoder 将src的 token 进化到 utterence 层面上：
        # TODO: 使用cls作为utterence的表示或者使用mean/max等表示（待测试）
        # b, src_utt_len, h
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        context_vec = self.ext_layer(sents_vec, mask_cls)
        # 补充实验
        # 直接句子分类效果：
        # return context_vec, mask_cls
        
        #########################################################
        # 使用类似BARTNER的方式进行decoder：
        dec_state = self.topic_decoder.init_decoder_state(sents_vec, context_vec)
        # decoder part:
        # 对tgt输入topic_tgt 进行修改；
        # topic_tgt : b, topic_sequence
        # 找哪些位置是tag token 哪些位置是src token 
        mapping_token_mask = topic_tgt.lt(self.src_start_index)  
        # ge: 大于等于 （将src部分mask掉）
        mapped_tokens = topic_tgt.masked_fill(topic_tgt.ge(self.src_start_index), 0)
        # 映射到在vocab中的位置： 99，99，99... tag_index,...,1
        # print(mapped_tokens)
        tag_mapped_tokens = self.mapping[mapped_tokens].to(mapped_tokens)
        tag_mapped_tokens = tag_mapped_tokens.masked_fill(~mask_topic_target, self.pad_token_id)
        # print(tag_mapped_tokens)
        # 找到src token的位置：
        src_tokens_index = topic_tgt - self.src_start_index # bsz x tgt_num
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        # print(src_tokens_index)
        # 从相应位置中提取utterence向量作为对应的decoder输入：
        for i in range(src_tokens_index.size(0)):
            assert int(src_tokens_index[i].max()) < len(sents_vec[i])
        utt_mapped_list = []
        for i in range(sents_vec.size(0)):
            utt_mapped_list.append(torch.index_select(sents_vec[i], 0, src_tokens_index[i]))
        # b, tgt_len, h
        utt_mapped_vec = torch.stack(utt_mapped_list).to(sents_vec)
        # tag embedding : b, tgt_len, emb_h
        tag_mapped_vec = self.topic_decoder.embeddings(tag_mapped_tokens)
        assert tag_mapped_vec.dim() == 3  # len x batch x embedding_dim 
        assert tag_mapped_vec.size(-1) == utt_mapped_vec.size(-1)
        # 将 utt tensor 与 tag tensor 拼接起来作为decoder的输入：
        # True : tag_mapped_tokens ; False : word_mapped_tokens
        decoder_input_list = []
        for i in range(mapping_token_mask.size(0)):
            condition = mapping_token_mask[i].unsqueeze(-1)
            # print(condition)
            decoder_input_list.append(torch.where(condition, tag_mapped_vec[i], utt_mapped_vec[i]))
        decoder_input = torch.stack(decoder_input_list).to(utt_mapped_vec)
        assert decoder_input.dim() == 3
        # 去掉最后一个eos token 然后输入decoder中：
        topic_output, state_topic = self.topic_decoder(decoder_input[:,:-1,:],context_vec,dec_state,memory_masks=~mask_cls,tgt_masks=mask_topic_target[:,:-1])
        # 对生成的 topic_output 进行处理得到并为后续计算loss
        # b, tgt, vocab_size(tag+2+cur_utt_num)
        logits = topic_output.new_full((topic_output.size(0), topic_output.size(1), self.src_start_index+sents_vec.size(1)),
                                       fill_value=-1e24)

        # 首先计算的是 end 标签以及 Tag 标签：
        eos_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[1:2]))  # bsz x max_len x 1
        pad_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[0:1]))
        # mapping_ids = torch.LongTensor(self.label_ids)
        # tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
        # tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[mapping_ids]))
        tag_scores = F.linear(topic_output, self.dropout_layer(self.topic_decoder.embeddings.weight[2:3]))

        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
        # b, utt_num, h
        if hasattr(self, 'encoder_mlp'):
            context_vec = self.encoder_mlp(context_vec)

        # 先把feature合并一下 
        sents_vec = self.dropout_layer(sents_vec)
        if avg_feature: 
            context_vec = (context_vec + sents_vec)/2
        # 计算utt hidden 与 decoder hidden 相关得分
        # b, tgt_len, utt_len 
        word_scores = torch.einsum('blh,bnh->bln', topic_output, context_vec)  # bsz x max_len x max_word_len
        if not avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', topic_output, sents_vec)  # bsz x max_len x max_word_len
            word_scores = (gen_scores + word_scores)/2
        
        # 针对最后一维即utt padding 进行mask
        mask = ~mask_cls.unsqueeze(1)
        word_scores = word_scores.masked_fill(mask, -1e32)

        # 最后整合到一起：
        # sos 位置空缺保证输出结果没有sos标签：
        logits[:, :, 0:1] = pad_scores
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        #TODO: 加上后面的decoder summary tgt token 模块（real utt mask 以及 提取出tgt tag部分作为监督信号）
        # 对tag hidden state 进行打乱重排：(aux task)
        # Training part:
        device = topic_output.device
        dim = topic_output.size(-1)
        tag_mask = topic_tgt.eq(self.tag_index)[:,:-1]
        # tag tgt index(containing eos)
        tag_num = tag_mask.sum(1)
        tag_tgt_list = []
        tag_src_mask_list = []
        reshuffle_list = []
        for i in range(tag_num.size(0)):
            shuffle_index = [i for i in range(int(tag_num[i]))]
            np.random.shuffle(shuffle_index)
            # shuffle and reshuffle 
            tag_tgt_list.append(torch.cat([torch.arange(1,int(tag_num[i])+1)[shuffle_index],torch.tensor([0])]))
            reshuffle_list.append(np.argsort(shuffle_index))
            tag_src_mask_list.append(torch.ones(int(tag_num[i])))
        # b, max_tgt_num+1
        tag_tgt = pad_sequence(tag_tgt_list,batch_first=True,padding_value=-1).to(device)
        tag_src_mask = pad_sequence(tag_src_mask_list,batch_first=True,padding_value=0).bool().to(device)
        tag_tgt_mask = ~(tag_tgt == -1)
        tag_tgt[tag_tgt == -1] = 0
        # tag_tgt_mask = pad_sequence(tag_tgt_mask_list,batch_first=True,padding_value=0)
        # 选取相对应的hidden state
        #TODO:根据index进行shuffle
        tag_list = []
        shuffle_tag_list = []
        for i in range(topic_output.size(0)):
            tag_list.append(torch.masked_select(topic_output[i],tag_mask[i].unsqueeze(-1)).reshape(-1,dim))
            shuffle_tag_list.append(torch.masked_select(topic_output[i],tag_mask[i].unsqueeze(-1)).reshape(-1,dim)[reshuffle_list[i]])
        # b, max_tag_num, h
        # print(tag_list)
        tag_vec = pad_sequence(tag_list,batch_first=True,padding_value=0).to(device)
        re_tag_vec = pad_sequence(shuffle_tag_list,batch_first=True,padding_value=0).to(device)
        # print(tag_vec.size())
        tg_decoder_input = torch.cat([self.pn_init_token.unsqueeze(0).expand(topic_output.size(0), 1, -1), tag_vec], 1).to(device)
        # tag reranker:
        tag_state = self.tag_decoder.init_decoder_state(tag_vec, tag_vec)
        assert tg_decoder_input.size(1) == tag_tgt_mask.size(1)
        tag_output, _ = self.tag_decoder(tg_decoder_input,re_tag_vec,tag_state,memory_masks=~tag_src_mask,tgt_masks=~tag_tgt_mask)
        # 计算tgt logits:
        tag_src_vec = torch.cat([self.terminate_state.unsqueeze(0).expand(topic_output.size(0), 1, -1), re_tag_vec], 1)
        # b, tag_tgt+sos, tag_tgt+eos
        tag_rank_scores = torch.einsum('blh,bnh->bln', tag_output, tag_src_vec)
        mask = ~tag_tgt_mask.unsqueeze(1)
        rank_logits = tag_rank_scores.masked_fill(mask, -1e32)

        return logits, rank_logits, tag_tgt, tag_tgt_mask

        # TODO: 将最终的解码部分和上面的全部部分拼接起来：(联合学习部分)
        
        #针对将user 与 agent进行统一操作：
        #在tgt中区分 user与agent 的mask:

        tgt_user_mask = tgt_fin_role[:,:-1].eq(self.user_tag_id)
        tgt_agent_mask = tgt_fin_role[:,:-1].eq(self.agent_tag_id)
        
        # 解码调用：
        tgt_dec_state = self.tog_decoder.init_decoder_state(src, top_vec)
        # 加上real utt mask(生成subsummary所关注的utt)
        outputs, state, user_scores, agent_scores, last_layer_score = self.tog_decoder(tgts[:, :-1],top_vec,tgt_dec_state,role_mask,tgt_user_mask,tgt_agent_mask,src_tgt_mask_final)
        
        batch, tgt_len, src_len = src.size(0) ,tgt_user_mask.size(1), src.size(1)

        tgt_user_mask = tgt_user_mask.unsqueeze(2).expand(batch, tgt_len, src_len)
        tgt_agent_mask = tgt_agent_mask.unsqueeze(2).expand(batch, tgt_len, src_len)

        #针对context utt hidden state 进行pointer-net generator的编写：
        # 使用和不使用 context utt layer 对应的 p-n generator 不同：
        # 对于 together decoding 的 PGN:
        # 首先计算每一步的src 关注分布与 summary hidden output 拼接：b, tgt, h*3
        # print(last_layer_score[0].size())
        # output_cat = torch.cat((torch.bmm(last_layer_score[0],top_vec),torch.bmm(last_layer_score[1],top_vec), outputs),-1)
        output_cat = torch.cat((torch.bmm(last_layer_score[0],top_vec), outputs),-1)
        
        return logits, rank_logits, tag_tgt, tag_tgt_mask, [outputs,output_cat], [user_scores, agent_scores, tgt_user_mask, tgt_agent_mask], last_layer_score
        
        ##############################################################################
        
        
        # return logits
        # user_dec_state = self.user_decoder.init_decoder_state(src, top_vec)
        # agent_dec_state = self.agent_decoder.init_decoder_state(src, top_vec)
        # print(tgts[0][:, :-1])
        # user_outputs, agent_outputs, state_user, state_agent, user_scores, agent_scores, kl_mask_user, kl_mask_agent = self.role_decoder(tgts[0][:, :-1], tgts[1][:, :-1], top_vec, user_dec_state, agent_dec_state, merge_type, inter_weight, role_mask, src_tgt_mask_final)
        # return [user_outputs, agent_outputs], None, [user_scores, agent_scores, kl_mask_user, kl_mask_agent]