# import os

# import numpy as np
# import torch
# import torch.nn as nn
# from tensorboardX import SummaryWriter

# import distributed
# from models.reporter import ReportMgr, Statistics
# from others.logging import logger
# from others.utils import test_rouge, rouge_results_to_str


# def _tally_parameters(model):
#     n_params = sum([p.nelement() for p in model.parameters()])
#     return n_params


# def build_trainer(args, device_id, model, optims,loss, tokenizer):
#     """
#     Simplify `Trainer` creation based on user `opt`s*
#     Args:
#         opt (:obj:`Namespace`): user options (usually from argument parsing)
#         model (:obj:`onmt.models.NMTModel`): the model to train
#         fields (dict): dict of fields
#         optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
#         data_type (str): string describing the type of data
#             e.g. "text", "img", "audio"
#         model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
#             used to save the model
#     """
#     device = "cpu" if args.visible_gpus == '-1' else "cuda"


#     grad_accum_count = args.accum_count
#     n_gpu = args.world_size

#     if device_id >= 0:
#         gpu_rank = int(args.gpu_ranks[device_id])
#     else:
#         gpu_rank = 0
#         n_gpu = 0

#     print('gpu_rank %d' % gpu_rank)

#     tensorboard_log_dir = args.model_path

#     writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

#     report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)


#     trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager, tokenizer)

#     # print(tr)
#     if (model):
#         n_params = _tally_parameters(model)
#         logger.info('* number of parameters: %d' % n_params)

#     return trainer


# class Trainer(object):
#     """
#     Class that controls the training process.

#     Args:
#             model(:py:class:`onmt.models.model.NMTModel`): translation model
#                 to train
#             train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
#                training loss computation
#             valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
#                training loss computation
#             optim(:obj:`onmt.utils.optimizers.Optimizer`):
#                the optimizer responsible for update
#             trunc_size(int): length of truncated back propagation through time
#             shard_size(int): compute loss in shards of this size for efficiency
#             data_type(string): type of the source input: [text|img|audio]
#             norm_method(string): normalization methods: [sents|tokens]
#             grad_accum_count(int): accumulate gradients this many times.
#             report_manager(:obj:`onmt.utils.ReportMgrBase`):
#                 the object that creates reports, or None
#             model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
#                 used to save a checkpoint.
#                 Thus nothing will be saved if this parameter is None
#     """

#     def __init__(self,  args, model,  optims, loss,
#                   grad_accum_count=1, n_gpu=1, gpu_rank=1,
#                   report_manager=None, tokenizer=None):
#         # Basic attributes.
#         self.args = args
#         self.save_checkpoint_steps = args.save_checkpoint_steps
#         self.model = model
#         self.optims = optims
#         self.grad_accum_count = grad_accum_count
#         self.n_gpu = n_gpu
#         self.gpu_rank = gpu_rank
#         self.report_manager = report_manager
#         self.tokenizer = tokenizer

#         self.loss = loss
#         self.logsoftmax = nn.LogSoftmax(dim=-1)
#         self.label_criterion = torch.nn.CrossEntropyLoss(reduction='none')

#         assert grad_accum_count > 0
#         # Set model in training mode.
#         if (model):
#             self.model.train()

#     def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
#         """
#         The main training loops.
#         by iterating over training data (i.e. `train_iter_fct`)
#         and running validation (i.e. iterating over `valid_iter_fct`

#         Args:
#             train_iter_fct(function): a function that returns the train
#                 iterator. e.g. something like
#                 train_iter_fct = lambda: generator(*args, **kwargs)
#             valid_iter_fct(function): same as train_iter_fct, for valid data
#             train_steps(int):
#             valid_steps(int):
#             save_checkpoint_steps(int):

#         Return:
#             None
#         """
#         logger.info('Start training...')

#         # step =  self.optim._step + 1
#         step =  self.optims[0]._step + 1
#         # step =  self.optims[1]._step + 1
#         step = 1
#         # print(step)

#         true_batchs = []
#         accum = 0
#         normalization = [0, 0, 0]
#         train_iter = train_iter_fct()

#         total_stats = [Statistics(), Statistics()]
#         report_stats = [Statistics(), Statistics(),Statistics()]
#         self._start_report_manager(start_time=total_stats[0].start_time)

#         while step <= train_steps:

#             reduce_counter = 0
#             for i, batch in enumerate(train_iter):
#                 if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

#                     true_batchs.append(batch)
#                     # num_tokens = batch.tgt_user[:, 1:].ne(self.loss.padding_idx).sum()
#                     # normalization[0] += num_tokens.item()
#                     # num_tokens = batch.tgt_agent[:, 1:].ne(self.loss.padding_idx).sum()
#                     # normalization[1] += num_tokens.item()
#                     # 关于topic tgt真实token个数：
#                     num_tokens = batch.mask_topic_target[:, 1:].sum()
#                     normalization[2] += num_tokens.item()

#                     num_tags = batch.topic_target.eq(2).sum()
#                     normalization[1] += num_tags.item()

#                     accum += 1
#                     if accum == self.grad_accum_count:
#                         reduce_counter += 1
#                         if self.n_gpu > 1:
#                             for i in range(3):
#                                 normalization[i] = sum(distributed
#                                                     .all_gather_list
#                                                     (normalization[i]))

#                         self._gradient_accumulation(
#                             true_batchs, normalization, total_stats,
#                             report_stats)

#                         report_stats[2] = self._maybe_report_training(
#                                 step, train_steps,
#                                 self.optims[0].learning_rate,
#                                 report_stats[2])
                        
#                         report_stats[1] = self._maybe_report_training(
#                                 step, train_steps,
#                                 self.optims[0].learning_rate,
#                                 report_stats[1])
#                         # for i in range(2):
#                         #     report_stats[i] = self._maybe_report_training(
#                         #         step, train_steps,
#                         #         self.optims[0].learning_rate,
#                         #         report_stats[i])

#                         true_batchs = []
#                         accum = 0
#                         normalization = [0, 0, 0]
#                         if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
#                             self._save(step)

#                         step += 1
#                         if step > train_steps:
#                             break
#             train_iter = train_iter_fct()

#         return total_stats

#     def validate(self, valid_iter, step=0):
#         """ Validate model.
#             valid_iter: validate data iterator
#         Returns:
#             :obj:`nmt.Statistics`: validation loss statistics
#         """
#         # Set model in validating mode.
#         self.model.eval()
#         stats = [Statistics(), Statistics(), Statistics()]

#         with torch.no_grad():
#             for batch in valid_iter:
#                 src = batch.src
#                 tgt_user = batch.tgt_user
#                 tgt_agent = batch.tgt_agent
#                 segs = batch.segs
#                 clss = batch.clss
#                 mask_src = batch.mask_src
#                 mask_tgt_user = batch.mask_tgt_user
#                 mask_tgt_agent = batch.mask_tgt_agent
#                 mask_cls = batch.mask_cls
#                 role_mask = batch.role_mask
#                 tgts = [tgt_user, tgt_agent]
#                 mask_tgts = [mask_tgt_user, mask_tgt_agent]

#                 tgt_user_seg_idxs = batch.tgt_user_seg_idxs
#                 mask_tgt_user_seg_idxs = batch.mask_tgt_user_seg_idxs
#                 tgt_agent_seg_idxs = batch.tgt_agent_seg_idxs
#                 mask_tgt_agent_seg_idxs = batch.mask_tgt_agent_seg_idxs

#                 src_lens = batch.src_lens
#                 mask_src_lens = batch.mask_src_lens
#                 tgt_user_lens = batch.tgt_user_lens
#                 mask_tgt_user_lens = batch.mask_tgt_user_lens
#                 tgt_agent_lens = batch.tgt_agent_lens
#                 mask_tgt_agent_lens = batch.mask_tgt_agent_lens

#                 user_utterances = batch.user_utterances
#                 agent_utterances = batch.agent_utterances
#                 mask_user_utterances = batch.mask_user_utterances
#                 mask_agent_utterances = batch.mask_agent_utterances

#                 user_src_tgt_mask_final = batch.user_src_tgt_mask_final
#                 agent_src_tgt_mask_final = batch.agent_src_tgt_mask_final

#                 tgt_seg_idxs = [tgt_user_seg_idxs, tgt_agent_seg_idxs]
#                 mask_tgt_seg_idxs = [mask_tgt_user_seg_idxs, mask_tgt_agent_seg_idxs]
#                 tgt_lens = [tgt_user_lens, tgt_agent_lens]
#                 mask_tgt_lens = [mask_tgt_user_lens, mask_tgt_agent_lens]
#                 gold_utterances = [user_utterances, agent_utterances]
#                 mask_utterances = [mask_user_utterances, mask_agent_utterances]
#                 src_tgt_mask_final = [user_src_tgt_mask_final, agent_src_tgt_mask_final]

#                 # outputs, _, _, _ = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens,mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final)
#                 # 模型调用：
#                 topic_tgt = batch.topic_target
#                 mask_topic_target = batch.mask_topic_target
#                 batch_size = topic_tgt.size(0)
#                 # b, tgt_l, vocab_size
#                 # outputs = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
#                 outputs, rank_logits, tag_tgt, tag_tgt_mask = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
#                 # loss 函数调用计算以及指标展示：
#                 assert outputs.size(1) == topic_tgt.size(1)-1 == mask_topic_target.size(1)-1
#                 # 针对 tag 标签位置进行weight：
#                 loss = self.label_criterion(outputs.contiguous().view(-1, outputs.size(-1)), topic_tgt[:,1:].contiguous().view(-1))
#                 loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
#                 # print(outputs)
#                 loss = (loss * mask_topic_target[:,1:].float().contiguous().view(-1)).sum()
#                 loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
#                 # print(loss)
#                 # 取mean 精确到非padding具体tgt个数：
#                 # print(float(normalization[2]))
#                 # loss = loss.div(float(normalization[2]))
#                 # 展示结果, 展示loss值和 Accuracy
#                 pred = self.logsoftmax(outputs).view(-1, outputs.size(2)).max(1)[1]
#                 gold = topic_tgt[:,1:].contiguous().view(-1).to(pred.device)
#                 # print(pred,gold)
#                 ## 打印出来看一下结果：
#                 gold_index = gold.reshape(batch_size,-1).cpu().numpy()
#                 pred_index = pred.reshape(batch_size,-1).cpu().numpy()
#                 for i in range(batch_size):
#                     non_pad = int(mask_topic_target[i,1:].sum())
#                     print(pred_index[i][:non_pad])
#                     print(gold_index[i][:non_pad])

#                 # 看一下找到topic的准确率和召回率：

#                 # pred_tag_mask = pred.eq(2)
#                 # gold_tag_mask = gold.eq(2)
#                 # tag_correct = pred_tag_mask & gold_tag_mask
#                 # stats_F = Statistics(loss.clone().item(), gold_tag_mask.sum().item(), tag_correct.sum().item())
#                 # stats_P = Statistics(loss.clone().item(), pred_tag_mask.sum().item(), tag_correct.sum().item())
#                 # stats[0].update(stats_F)
#                 # stats[1].update(stats_P)

#                 non_padding = mask_topic_target[:,1:].contiguous().view(-1).to(pred.device)
#                 # 计算Accuracy：
#                 # print(pred.eq(gold).size(),non_padding.size())
#                 # assert gold.size() == non_padding.size(), '#################'
#                 correct = pred.eq(gold)
#                 assert correct.size() == non_padding.size()
#                 num_correct = correct.masked_select(non_padding).sum().item()
#                 stats_ = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
#                 # batch_stats = self.loss.monolithic_compute_loss(batch, outputs)

#                 stats[2].update(stats_)
#                 # for i in range(2):
#                 #     stats[i].update(batch_stats[i])
#                 # Ranker part:
#                 # 展示结果, 展示loss值和 Accuracy
#                 pred = self.logsoftmax(rank_logits).view(-1, rank_logits.size(2)).max(1)[1]
#                 # print(pred)
#                 gold = tag_tgt.contiguous().view(-1).to(pred.device)
#                 ## 打印出来看一下结果：
#                 gold_index = gold.reshape(batch_size,-1).cpu().numpy()
#                 pred_index = pred.reshape(batch_size,-1).cpu().numpy()
#                 for i in range(batch_size):
#                     non_pad = int(tag_tgt_mask.sum())
#                     print(pred_index[i][:non_pad])
#                     print(gold_index[i][:non_pad])

#                 non_padding = tag_tgt_mask.contiguous().view(-1).to(pred.device)
#                 # 计算Accuracy：
#                 # print(pred.eq(gold).size(),non_padding.size())
#                 # assert gold.size() == non_padding.size(), '#################'
#                 correct = pred.eq(gold)
#                 assert correct.size() == non_padding.size()
#                 num_correct = correct.masked_select(non_padding).sum().item()
#                 stats_ = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
#                 stats[1].update(stats_)
#             # type = ['user', 'agent']
#             # for i in range(2):
#             #     logger.info('Type: %s' % type[i])
#             #     self._report_step(0, step, valid_stats=stats[i])
#             logger.info('Type: %s' % 'Topic')
#             # for i in range(3):
#             #     self._report_step(0, step, valid_stats=stats[i])
#             self._report_step(0, step, valid_stats=stats[2])
#             self._report_step(0, step, valid_stats=stats[1])
#             return stats


#     def _gradient_accumulation(self, true_batchs, normalization, total_stats,
#                                report_stats):
#         if self.grad_accum_count > 1:
#             self.model.zero_grad()

#         for batch in true_batchs:
#             if self.grad_accum_count == 1:
#                 self.model.zero_grad()

#             src = batch.src
#             tgt_user = batch.tgt_user
#             tgt_agent = batch.tgt_agent
#             segs = batch.segs
#             clss = batch.clss
#             mask_src = batch.mask_src
#             mask_tgt_user = batch.mask_tgt_user
#             mask_tgt_agent = batch.mask_tgt_agent
#             mask_cls = batch.mask_cls
#             role_mask = batch.role_mask

#             tgt_user_seg_idxs = batch.tgt_user_seg_idxs
#             mask_tgt_user_seg_idxs = batch.mask_tgt_user_seg_idxs
#             tgt_agent_seg_idxs = batch.tgt_agent_seg_idxs
#             mask_tgt_agent_seg_idxs = batch.mask_tgt_agent_seg_idxs

#             src_lens = batch.src_lens
#             mask_src_lens = batch.mask_src_lens
#             tgt_user_lens = batch.tgt_user_lens
#             mask_tgt_user_lens = batch.mask_tgt_user_lens
#             tgt_agent_lens = batch.tgt_agent_lens
#             mask_tgt_agent_lens = batch.mask_tgt_agent_lens

#             user_utterances = batch.user_utterances
#             agent_utterances = batch.agent_utterances
#             mask_user_utterances = batch.mask_user_utterances
#             mask_agent_utterances = batch.mask_agent_utterances

#             user_src_tgt_mask_final = batch.user_src_tgt_mask_final
#             agent_src_tgt_mask_final = batch.agent_src_tgt_mask_final
            
            
#             # print(src, tgt_final, tgt_user, tgt_agent)
#             # src_sents = self.tokenizer.convert_ids_to_tokens(src[0].cpu().numpy().tolist())
#             # tgt_final_sents = self.tokenizer.convert_ids_to_tokens(tgt_final[0].cpu().numpy().tolist())
#             # tgt_user_sents = self.tokenizer.convert_ids_to_tokens(tgt_user[0].cpu().numpy().tolist())
#             # tgt_agent_sents = self.tokenizer.convert_ids_to_tokens(tgt_agent[0].cpu().numpy().tolist())
#             # print(src_sents, tgt_final_sents, tgt_user_sents, tgt_agent_sents)
#             # exit()
#             tgts = [tgt_user, tgt_agent]
#             mask_tgts = [mask_tgt_user, mask_tgt_agent]
#             tgt_seg_idxs = [tgt_user_seg_idxs, tgt_agent_seg_idxs]
#             mask_tgt_seg_idxs = [mask_tgt_user_seg_idxs, mask_tgt_agent_seg_idxs]
#             tgt_lens = [tgt_user_lens, tgt_agent_lens]
#             mask_tgt_lens = [mask_tgt_user_lens, mask_tgt_agent_lens]
#             gold_utterances = [user_utterances, agent_utterances]
#             mask_utterances = [mask_user_utterances, mask_agent_utterances]
#             src_tgt_mask_final = [user_src_tgt_mask_final, agent_src_tgt_mask_final]

#             # outputs, _, scores, utt_scores = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens,mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final)
#             # outputs, _, scores = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, src_tgt_mask_final)
#             # 模型调用：
#             topic_tgt = batch.topic_target
#             mask_topic_target = batch.mask_topic_target
#             # batch_size = topic_tgt.size(0)
#             # b, tgt_l, vocab_size
#             # rank_logits: b, tag_tgt+sos, tag_tgt+eos
#             # outputs = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
#             outputs, rank_logits, tag_tgt, tag_tgt_mask = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
#             # loss 函数调用计算以及指标展示：
#             assert outputs.size(1) == topic_tgt.size(1)-1 == mask_topic_target.size(1)-1
#             # # 针对 tag 标签位置进行weight：
#             # weight = torch.ones(outputs.size(-1))
#             # weight[2:213] = 3
#             # weight = weight.to(outputs)
#             # self.label_criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
#             loss = self.label_criterion(outputs.contiguous().view(-1, outputs.size(-1)), topic_tgt[:,1:].contiguous().view(-1))
#             loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
#             # print(outputs)
#             loss = (loss * mask_topic_target[:,1:].float().contiguous().view(-1)).sum()
#             loss_tag = (loss_tag * tag_tgt_mask.float().contiguous().view(-1)).sum()
#             # print(loss)
#             # 取mean 精确到非padding具体tgt个数：
#             # print(float(normalization[2]))
#             # loss = loss.div(float(normalization[2]))
#             # 展示结果, 展示loss值和 Accuracy
#             pred = self.logsoftmax(outputs).view(-1, outputs.size(2)).max(1)[1]
#             # print(pred)
#             gold = topic_tgt[:,1:].contiguous().view(-1).to(pred.device)
#             non_padding = mask_topic_target[:,1:].contiguous().view(-1).to(pred.device)
#             # 计算Accuracy：
#             # print(pred.eq(gold).size(),non_padding.size())
#             # assert gold.size() == non_padding.size(), '#################'
#             correct = pred.eq(gold)
#             assert correct.size() == non_padding.size()
#             num_correct = correct.masked_select(non_padding).sum().item()
#             stats = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
#             report_stats[2].update(stats)
#             # Ranker part:
#             # 展示结果, 展示loss值和 Accuracy
#             pred = self.logsoftmax(rank_logits).view(-1, rank_logits.size(2)).max(1)[1]
#             # print(pred)
#             gold = tag_tgt.contiguous().view(-1).to(pred.device)
#             non_padding = tag_tgt_mask.contiguous().view(-1).to(pred.device)
#             # 计算Accuracy：
#             # print(pred.eq(gold).size(),non_padding.size())
#             # assert gold.size() == non_padding.size(), '#################'
#             correct = pred.eq(gold)
#             assert correct.size() == non_padding.size()
#             num_correct = correct.masked_select(non_padding).sum().item()
#             stats = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
#             report_stats[1].update(stats)

#             # 反向传播
#             loss = loss.div(float(normalization[2])) + 0.5*loss_tag.div(float(normalization[1]))
#             # loss = loss.div(float(normalization[2]))
#             loss.backward()
#             # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores)
#             # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores, utt_scores, gold_utterances, mask_utterances)
#             # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores)
#             #batch_stats = self.loss.sharded_compute_loss(batch, outputs, None, normalization)
#             # for i in range(2):
#             #     batch_stats[i].n_docs = int(src.size(0))
#             #     total_stats[i].update(batch_stats[i])
#             #     report_stats[i].update(batch_stats[i])

#             # 4. Update the parameters and statistics.
#             if self.grad_accum_count == 1:
#                 # Multi GPU gradient gather
#                 if self.n_gpu > 1:
#                     grads = [p.grad.data for p in self.model.parameters()
#                              if p.requires_grad
#                              and p.grad is not None]
#                     distributed.all_reduce_and_rescale_tensors(
#                         grads, float(1))

#                 for o in self.optims:
#                     o.step()
#         #exit()

#         # in case of multi step gradient accumulation,
#         # update only after accum batches
#         if self.grad_accum_count > 1:
#             if self.n_gpu > 1:
#                 grads = [p.grad.data for p in self.model.parameters()
#                          if p.requires_grad
#                          and p.grad is not None]
#                 distributed.all_reduce_and_rescale_tensors(
#                     grads, float(1))
#             for o in self.optims:
#                 o.step()

#     # 对比试验
#     def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
#         """ Validate model.
#             valid_iter: validate data iterator
#         Returns:
#             :obj:`nmt.Statistics`: validation loss statistics
#         """
#         # Set model in validating mode.
#         def _get_ngrams(n, text):
#             ngram_set = set()
#             text_length = len(text)
#             max_index_ngram_start = text_length - n
#             for i in range(max_index_ngram_start + 1):
#                 ngram_set.add(tuple(text[i:i + n]))
#             return ngram_set

#         def _block_tri(c, p):
#             tri_c = _get_ngrams(3, c.split())
#             for s in p:
#                 tri_s = _get_ngrams(3, s.split())
#                 if len(tri_c.intersection(tri_s))>0:
#                     return True
#             return False

#         if (not cal_lead and not cal_oracle):
#             self.model.eval()
#         stats = Statistics()

#         can_path = '%s_step%d.candidate'%(self.args.result_path,step)
#         gold_path = '%s_step%d.gold' % (self.args.result_path, step)
#         with open(can_path, 'w') as save_pred:
#             with open(gold_path, 'w') as save_gold:
#                 with torch.no_grad():
#                     for batch in test_iter:
#                         gold = []
#                         pred = []
#                         if (cal_lead):
#                             selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
#                         for i, idx in enumerate(selected_ids):
#                             _pred = []
#                             if(len(batch.src_str[i])==0):
#                                 continue
#                             for j in selected_ids[i][:len(batch.src_str[i])]:
#                                 if(j>=len( batch.src_str[i])):
#                                     continue
#                                 candidate = batch.src_str[i][j].strip()
#                                 _pred.append(candidate)

#                                 if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
#                                     break

#                             _pred = '<q>'.join(_pred)
#                             if(self.args.recall_eval):
#                                 _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

#                             pred.append(_pred)
#                             gold.append(batch.tgt_str[i])

#                         for i in range(len(gold)):
#                             save_gold.write(gold[i].strip()+'\n')
#                         for i in range(len(pred)):
#                             save_pred.write(pred[i].strip()+'\n')
#         if(step!=-1 and self.args.report_rouge):
#             rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
#             logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
#         self._report_step(0, step, valid_stats=stats)

#         return stats

#     def _save(self, step):
#         real_model = self.model
#         # real_generator = (self.generator.module
#         #                   if isinstance(self.generator, torch.nn.DataParallel)
#         #                   else self.generator)

#         model_state_dict = real_model.state_dict()
#         # generator_state_dict = real_generator.state_dict()
#         checkpoint = {
#             'model': model_state_dict,
#             # 'generator': generator_state_dict,
#             'opt': self.args,
#             'optims': self.optims,
#         }
#         checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
#         logger.info("Saving checkpoint %s" % checkpoint_path)
#         # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
#         if (not os.path.exists(checkpoint_path)):
#             torch.save(checkpoint, checkpoint_path)
#             return checkpoint, checkpoint_path

#     def _start_report_manager(self, start_time=None):
#         """
#         Simple function to start report manager (if any)
#         """
#         if self.report_manager is not None:
#             if start_time is None:
#                 self.report_manager.start()
#             else:
#                 self.report_manager.start_time = start_time

#     def _maybe_gather_stats(self, stat):
#         """
#         Gather statistics in multi-processes cases

#         Args:
#             stat(:obj:onmt.utils.Statistics): a Statistics object to gather
#                 or None (it returns None in this case)

#         Returns:
#             stat: the updated (or unchanged) stat object
#         """
#         if stat is not None and self.n_gpu > 1:
#             return Statistics.all_gather_stats(stat)
#         return stat

#     def _maybe_report_training(self, step, num_steps, learning_rate,
#                                report_stats):
#         """
#         Simple function to report training stats (if report_manager is set)
#         see `onmt.utils.ReportManagerBase.report_training` for doc
#         """
#         if self.report_manager is not None:
#             return self.report_manager.report_training(
#                 step, num_steps, learning_rate, report_stats,
#                 multigpu=self.n_gpu > 1)

#     def _report_step(self, learning_rate, step, train_stats=None,
#                      valid_stats=None):
#         """
#         Simple function to report stats (if report_manager is set)
#         see `onmt.utils.ReportManagerBase.report_step` for doc
#         """
#         if self.report_manager is not None:
#             return self.report_manager.report_step(
#                 learning_rate, step, train_stats=train_stats,
#                 valid_stats=valid_stats)

#     def _maybe_save(self, step):
#         """
#         Save the model if a model saver is set
#         """
#         if self.model_saver is not None:
#             self.model_saver.maybe_save(step)
import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import distributed
from models.reporter import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims,loss, tokenizer):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)


    trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager, tokenizer)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optims, loss,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None, tokenizer=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.tokenizer = tokenizer

        self.loss = loss
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.label_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step =  self.optims[0]._step + 1
        # step =  self.optims[1]._step + 1
        step = 1
        # print(step)

        true_batchs = []
        accum = 0
        normalization = [0, 0, 0]
        train_iter = train_iter_fct()

        total_stats = [Statistics(), Statistics()]
        report_stats = [Statistics(), Statistics(),Statistics()]
        self._start_report_manager(start_time=total_stats[0].start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    # num_tokens = batch.tgt_user[:, 1:].ne(self.loss.padding_idx).sum()
                    # normalization[0] += num_tokens.item()
                    # num_tokens = batch.tgt_agent[:, 1:].ne(self.loss.padding_idx).sum()
                    # normalization[1] += num_tokens.item()
                    # 关于topic tgt真实token个数：

                    # num_tokens = batch.mask_cls.eq(1).sum()
                    # normalization[2] += num_tokens.item()

                    num_tokens = batch.mask_topic_target[:, 1:].sum()
                    normalization[2] += num_tokens.item()

                    num_tags = batch.topic_target.eq(2).sum()
                    normalization[1] += num_tags.item()

                    # # summary part:
                    # num_tgt_tokens = batch.tgt_all[:, 1:].ne(self.loss.padding_idx).sum()
                    # normalization[0] += num_tgt_tokens.item()
                    

                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            for i in range(3):
                                normalization[i] = sum(distributed
                                                    .all_gather_list
                                                    (normalization[i]))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats[2] = self._maybe_report_training(
                                step, train_steps,
                                self.optims[0].learning_rate,
                                report_stats[2])
                        
                        # report_stats[1] = self._maybe_report_training(
                        #         step, train_steps,
                        #         self.optims[0].learning_rate,
                        #         report_stats[1])

                        # for i in range(3):
                        #     report_stats[i] = self._maybe_report_training(
                        #         step, train_steps,
                        #         self.optims[0].learning_rate,
                        #         report_stats[i])

                        true_batchs = []
                        accum = 0
                        normalization = [0, 0, 0]
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = [Statistics(), Statistics(), Statistics()]

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                tgt_user = batch.tgt_user
                tgt_agent = batch.tgt_agent
                segs = batch.segs
                clss = batch.clss
                mask_src = batch.mask_src
                mask_tgt_user = batch.mask_tgt_user
                mask_tgt_agent = batch.mask_tgt_agent
                mask_cls = batch.mask_cls
                role_mask = batch.role_mask
                tgts = [tgt_user, tgt_agent]
                mask_tgts = [mask_tgt_user, mask_tgt_agent]

                tgt_user_seg_idxs = batch.tgt_user_seg_idxs
                mask_tgt_user_seg_idxs = batch.mask_tgt_user_seg_idxs
                tgt_agent_seg_idxs = batch.tgt_agent_seg_idxs
                mask_tgt_agent_seg_idxs = batch.mask_tgt_agent_seg_idxs

                src_lens = batch.src_lens
                mask_src_lens = batch.mask_src_lens
                tgt_user_lens = batch.tgt_user_lens
                mask_tgt_user_lens = batch.mask_tgt_user_lens
                tgt_agent_lens = batch.tgt_agent_lens
                mask_tgt_agent_lens = batch.mask_tgt_agent_lens

                user_utterances = batch.user_utterances
                agent_utterances = batch.agent_utterances
                mask_user_utterances = batch.mask_user_utterances
                mask_agent_utterances = batch.mask_agent_utterances

                user_src_tgt_mask_final = batch.user_src_tgt_mask_final
                agent_src_tgt_mask_final = batch.agent_src_tgt_mask_final

                tgt_seg_idxs = [tgt_user_seg_idxs, tgt_agent_seg_idxs]
                mask_tgt_seg_idxs = [mask_tgt_user_seg_idxs, mask_tgt_agent_seg_idxs]
                tgt_lens = [tgt_user_lens, tgt_agent_lens]
                mask_tgt_lens = [mask_tgt_user_lens, mask_tgt_agent_lens]
                gold_utterances = [user_utterances, agent_utterances]
                mask_utterances = [mask_user_utterances, mask_agent_utterances]
                src_tgt_mask_final = [user_src_tgt_mask_final, agent_src_tgt_mask_final]

                # outputs, _, _, _ = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens,mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final)
                # 模型调用：
                topic_tgt = batch.topic_target
                mask_topic_target = batch.mask_topic_target
                batch_size = topic_tgt.size(0)
                # 调用total tgt :(所对应的dataloader中也要更新)
                # tgt_all = batch.tgt_all
                # tgt_fin_role = batch.tgt_fin_role
                # mask_tgt_all = batch.mask_tgt_all
                # src_tgt_mask_final_tog = batch.src_tgt_mask_final_tog

                # b, tgt_l, vocab_size
                # outputs = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
                # outputs, rank_logits, tag_tgt, tag_tgt_mask = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
                '''
                sent_scores, mask_cls = self.model(src, segs, clss, mask_src, mask_cls)

                src_len, dim = sent_scores.size(1), sent_scores.size(-1)

                loss = self.label_criterion(sent_scores.view(-1, dim), batch.intent_label.contiguous().view(-1))
                loss = (loss * mask_cls.float().view(-1)).sum()
            
                # 展示结果, 展示loss值和 Accuracy
                pred = self.logsoftmax(sent_scores).view(-1, sent_scores.size(2)).max(1)[1]
                # print(pred)
                gold = batch.intent_label.contiguous().view(-1).to(pred.device)

                non_padding = mask_cls.contiguous().view(-1).to(pred.device)
                # 计算Accuracy：
                # print(pred.eq(gold).size(),non_padding.size())
                # assert gold.size() == non_padding.size(), '#################'
                correct = pred.eq(gold)
                assert correct.size() == non_padding.size()
                num_correct = correct.masked_select(non_padding).sum().item()
                stats_ = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
                stats[2].update(stats_)
                '''
                #########################################
                
                # outputs, rank_logits, tag_tgt, tag_tgt_mask, s_outputs, scores, last_layer_score\
                # = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls,\
                #              tgt_all, role_mask, tgt_fin_role, src_tgt_mask_final_tog)
                outputs, rank_logits, tag_tgt, tag_tgt_mask = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
                # loss 函数调用计算以及指标展示：
                assert outputs.size(1) == topic_tgt.size(1)-1 == mask_topic_target.size(1)-1
                # 针对 tag 标签位置进行weight：
                loss = self.label_criterion(outputs.contiguous().view(-1, outputs.size(-1)), topic_tgt[:,1:].contiguous().view(-1))
                loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
                # print(outputs)
                loss = (loss * mask_topic_target[:,1:].float().contiguous().view(-1)).sum()
                loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
                # print(loss)
                # 取mean 精确到非padding具体tgt个数：
                # print(float(normalization[2]))
                # loss = loss.div(float(normalization[2]))
                # 展示结果, 展示loss值和 Accuracy
                pred = self.logsoftmax(outputs).view(-1, outputs.size(2)).max(1)[1]
                gold = topic_tgt[:,1:].contiguous().view(-1).to(pred.device)
                # print(pred,gold)
                ## 打印出来看一下结果：
                gold_index = gold.reshape(batch_size,-1).cpu().numpy()
                pred_index = pred.reshape(batch_size,-1).cpu().numpy()
                # for i in range(batch_size):
                #     non_pad = int(mask_topic_target[i,1:].sum())
                #     print(pred_index[i][:non_pad])
                #     print(gold_index[i][:non_pad])

                # 看一下找到topic的准确率和召回率：

                # pred_tag_mask = pred.eq(2)
                # gold_tag_mask = gold.eq(2)
                # tag_correct = pred_tag_mask & gold_tag_mask
                # stats_F = Statistics(loss.clone().item(), gold_tag_mask.sum().item(), tag_correct.sum().item())
                # stats_P = Statistics(loss.clone().item(), pred_tag_mask.sum().item(), tag_correct.sum().item())
                # stats[0].update(stats_F)
                # stats[1].update(stats_P)

                non_padding = mask_topic_target[:,1:].contiguous().view(-1).to(pred.device)
                # 计算Accuracy：
                # print(pred.eq(gold).size(),non_padding.size())
                # assert gold.size() == non_padding.size(), '#################'
                correct = pred.eq(gold)
                assert correct.size() == non_padding.size()
                num_correct = correct.masked_select(non_padding).sum().item()
                stats_ = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
                # batch_stats = self.loss.monolithic_compute_loss(batch, outputs)

                stats[2].update(stats_)
                # for i in range(2):
                #     stats[i].update(batch_stats[i])
                # Ranker part:
                # 展示结果, 展示loss值和 Accuracy
                pred = self.logsoftmax(rank_logits).view(-1, rank_logits.size(2)).max(1)[1]
                # print(pred)
                gold = tag_tgt.contiguous().view(-1).to(pred.device)
                ## 打印出来看一下结果：
                gold_index = gold.reshape(batch_size,-1).cpu().numpy()
                pred_index = pred.reshape(batch_size,-1).cpu().numpy()
                # for i in range(batch_size):
                #     non_pad = int(tag_tgt_mask.sum())
                #     print(pred_index[i][:non_pad])
                #     print(gold_index[i][:non_pad])

                non_padding = tag_tgt_mask.contiguous().view(-1).to(pred.device)
                # 计算Accuracy：
                # print(pred.eq(gold).size(),non_padding.size())
                # assert gold.size() == non_padding.size(), '#################'
                correct = pred.eq(gold)
                assert correct.size() == non_padding.size()
                num_correct = correct.masked_select(non_padding).sum().item()
                stats_ = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
                stats[1].update(stats_)
                
            ##########################################################
            # type = ['user', 'agent']
            # for i in range(2):
            #     logger.info('Type: %s' % type[i])
            #     self._report_step(0, step, valid_stats=stats[i])
            logger.info('Type: %s' % 'Topic')
            # for i in range(3):
            #     self._report_step(0, step, valid_stats=stats[i])
            self._report_step(0, step, valid_stats=stats[2])
            # self._report_step(0, step, valid_stats=stats[1])

            return stats


    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            tgt_user = batch.tgt_user
            tgt_agent = batch.tgt_agent
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_tgt_user = batch.mask_tgt_user
            mask_tgt_agent = batch.mask_tgt_agent
            mask_cls = batch.mask_cls
            role_mask = batch.role_mask

            tgt_user_seg_idxs = batch.tgt_user_seg_idxs
            mask_tgt_user_seg_idxs = batch.mask_tgt_user_seg_idxs
            tgt_agent_seg_idxs = batch.tgt_agent_seg_idxs
            mask_tgt_agent_seg_idxs = batch.mask_tgt_agent_seg_idxs

            src_lens = batch.src_lens
            mask_src_lens = batch.mask_src_lens
            tgt_user_lens = batch.tgt_user_lens
            mask_tgt_user_lens = batch.mask_tgt_user_lens
            tgt_agent_lens = batch.tgt_agent_lens
            mask_tgt_agent_lens = batch.mask_tgt_agent_lens

            user_utterances = batch.user_utterances
            agent_utterances = batch.agent_utterances
            mask_user_utterances = batch.mask_user_utterances
            mask_agent_utterances = batch.mask_agent_utterances

            user_src_tgt_mask_final = batch.user_src_tgt_mask_final
            agent_src_tgt_mask_final = batch.agent_src_tgt_mask_final
            
            
            # print(src, tgt_final, tgt_user, tgt_agent)
            # src_sents = self.tokenizer.convert_ids_to_tokens(src[0].cpu().numpy().tolist())
            # tgt_final_sents = self.tokenizer.convert_ids_to_tokens(tgt_final[0].cpu().numpy().tolist())
            # tgt_user_sents = self.tokenizer.convert_ids_to_tokens(tgt_user[0].cpu().numpy().tolist())
            # tgt_agent_sents = self.tokenizer.convert_ids_to_tokens(tgt_agent[0].cpu().numpy().tolist())
            # print(src_sents, tgt_final_sents, tgt_user_sents, tgt_agent_sents)
            # exit()
            tgts = [tgt_user, tgt_agent]
            mask_tgts = [mask_tgt_user, mask_tgt_agent]
            tgt_seg_idxs = [tgt_user_seg_idxs, tgt_agent_seg_idxs]
            mask_tgt_seg_idxs = [mask_tgt_user_seg_idxs, mask_tgt_agent_seg_idxs]
            tgt_lens = [tgt_user_lens, tgt_agent_lens]
            mask_tgt_lens = [mask_tgt_user_lens, mask_tgt_agent_lens]
            gold_utterances = [user_utterances, agent_utterances]
            mask_utterances = [mask_user_utterances, mask_agent_utterances]
            src_tgt_mask_final = [user_src_tgt_mask_final, agent_src_tgt_mask_final]

            # outputs, _, scores, utt_scores = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, tgt_seg_idxs, mask_tgt_seg_idxs, src_lens,mask_src_lens, tgt_lens, mask_tgt_lens, gold_utterances, mask_utterances, src_tgt_mask_final)
            # outputs, _, scores = self.model(src, tgts, segs, clss, mask_src, mask_tgts, mask_cls, self.args.merge, self.args.inter_weight, role_mask, src_tgt_mask_final)
            # 模型调用：
            topic_tgt = batch.topic_target
            mask_topic_target = batch.mask_topic_target
            # 调用total tgt :(所对应的dataloader中也要更新)
            # tgt_all = batch.tgt_all
            # tgt_fin_role = batch.tgt_fin_role
            # mask_tgt_all = batch.mask_tgt_all
            # src_tgt_mask_final_tog = batch.src_tgt_mask_final_tog
            # batch_size = topic_tgt.size(0)
            # b, tgt_l, vocab_size
            # rank_logits: b, tag_tgt+sos, tag_tgt+eos
            # outputs = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
            
            # context_scores: b, utt_num
            # sent_scores, mask_cls\
            #     = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls,\
            #                  tgt_all, role_mask, tgt_fin_role, src_tgt_mask_final_tog)
            '''
            sent_scores, mask_cls = self.model(src, segs, clss, mask_src, mask_cls)

            src_len, dim = sent_scores.size(1), sent_scores.size(-1)

            loss = self.label_criterion(sent_scores.view(-1, dim), batch.intent_label.contiguous().view(-1))
            loss = (loss * mask_cls.float().view(-1)).sum()

            # 展示结果, 展示loss值和 Accuracy
            pred = self.logsoftmax(sent_scores).view(-1, sent_scores.size(2)).max(1)[1]
            # print(pred)
            gold = batch.intent_label.contiguous().view(-1).to(pred.device)

            non_padding = mask_cls.contiguous().view(-1).to(pred.device)
            # 计算Accuracy：
            # print(pred.eq(gold).size(),non_padding.size())
            # assert gold.size() == non_padding.size(), '#################'
            correct = pred.eq(gold)
            assert correct.size() == non_padding.size()
            num_correct = correct.masked_select(non_padding).sum().item()
            stats = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
            report_stats[2].update(stats)
            loss = loss.div(float(normalization[2]))
            loss.backward()
            '''


            # (loss / loss.numel()).backward()
            ############################################################################

            # 整个模型的输出：
            # outputs, rank_logits, tag_tgt, tag_tgt_mask, s_outputs, scores, last_layer_score\
            #     = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls,\
            #                  tgt_all, role_mask, tgt_fin_role, src_tgt_mask_final_tog)

            outputs, rank_logits, tag_tgt, tag_tgt_mask = self.model(src, topic_tgt, segs, clss, mask_src, mask_topic_target, mask_cls)
            # loss 函数调用计算以及指标展示：
            assert outputs.size(1) == topic_tgt.size(1)-1 == mask_topic_target.size(1)-1
            # # 针对 tag 标签位置进行weight：
            # weight = torch.ones(outputs.size(-1))
            # weight[2:213] = 3
            # weight = weight.to(outputs)
            # self.label_criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
            loss = self.label_criterion(outputs.contiguous().view(-1, outputs.size(-1)), topic_tgt[:,1:].contiguous().view(-1))
            loss_tag = self.label_criterion(rank_logits.contiguous().view(-1, rank_logits.size(-1)), tag_tgt.contiguous().view(-1))
            # print(outputs)
            loss = (loss * mask_topic_target[:,1:].float().contiguous().view(-1)).sum()
            loss_tag = (loss_tag * tag_tgt_mask.float().contiguous().view(-1)).sum()
            # print(loss)
            # 取mean 精确到非padding具体tgt个数：
            # print(float(normalization[2]))
            # loss = loss.div(float(normalization[2]))
            # 展示结果, 展示loss值和 Accuracy
            pred = self.logsoftmax(outputs).view(-1, outputs.size(2)).max(1)[1]
            # print(pred)
            gold = topic_tgt[:,1:].contiguous().view(-1).to(pred.device)
            non_padding = mask_topic_target[:,1:].contiguous().view(-1).to(pred.device)
            # 计算Accuracy：
            # print(pred.eq(gold).size(),non_padding.size())
            # assert gold.size() == non_padding.size(), '#################'
            correct = pred.eq(gold)
            assert correct.size() == non_padding.size()
            num_correct = correct.masked_select(non_padding).sum().item()
            stats = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
            report_stats[2].update(stats)
            # Ranker part:
            # 展示结果, 展示loss值和 Accuracy
            pred = self.logsoftmax(rank_logits).view(-1, rank_logits.size(2)).max(1)[1]
            # print(pred)
            gold = tag_tgt.contiguous().view(-1).to(pred.device)
            non_padding = tag_tgt_mask.contiguous().view(-1).to(pred.device)
            # 计算Accuracy：
            # print(pred.eq(gold).size(),non_padding.size())
            # assert gold.size() == non_padding.size(), '#################'
            correct = pred.eq(gold)
            assert correct.size() == non_padding.size()
            num_correct = correct.masked_select(non_padding).sum().item()
            stats = Statistics(loss.clone().item(), non_padding.sum().item(), num_correct)
            report_stats[1].update(stats)

            # TODO:Summary Decoder 部分 loss：
            # batch_stats, s_loss = self.loss.sharded_compute_loss(batch, s_outputs, self.args.generator_shard_size, normalization[0], self.args, scores, last_layer_score)
            # report_stats[0].update(batch_stats[0])

            # 反向传播(设计按模块冻结)
            # loss = loss.div(float(normalization[2])) + 0.5*loss_tag.div(float(normalization[1])) + s_loss
            # loss = loss.div(float(normalization[2])) + 0.5*loss_tag.div(float(normalization[1])) + s_loss*5
            loss = loss.div(float(normalization[2]))
            loss.backward()

            ############################################################################
            # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores)
            # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores, utt_scores, gold_utterances, mask_utterances)
            # batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization, self.args, scores)
            #batch_stats = self.loss.sharded_compute_loss(batch, outputs, None, normalization)
            # for i in range(2):
            #     batch_stats[i].n_docs = int(src.size(0))
            #     total_stats[i].update(batch_stats[i])
            #     report_stats[i].update(batch_stats[i])

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))

                for o in self.optims:
                    o.step()
        #exit()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for o in self.optims:
                o.step()

    # 对比试验
    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        gold = []
                        pred = []
                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j>=len( batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
