#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer
from cal_rouge import cal_rouge_path


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        # self.change_token = symbols['AGENT']
        self.change_token = symbols['EOQ']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str_fin, batch.src
        # preds, pred_score, gold_score, u_str, a_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str_user,batch.tgt_str_agent, batch.src
        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##','')
            gold_sent = ' '.join(tgt_str[b].split())
            # gold_u, gold_a = ' '.join(u_str[b].split()), ' '.join(a_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents, gold_u, gold_a, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    # 用来构建topic sequence的生成器：
    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        # gold_u_path = self.args.result_path + '.%d.user.gold' % step
        # gold_a_path = self.args.result_path + '.%d.agent.gold' % step
        # can_u_path = self.args.result_path + '.%d.user.candidate' % step
        # can_a_path = self.args.result_path + '.%d.agent.candidate' % step

        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        # self.gold_u_path_file = codecs.open(gold_u_path, 'w', 'utf-8')
        # self.gold_a_path_file = codecs.open(gold_a_path, 'w', 'utf-8')
        # self.can_u_path_file = codecs.open(can_u_path, 'w', 'utf-8')
        # self.can_a_path_file = codecs.open(can_a_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # 加载生成的utt mask:
        utt_mask = []
        utt_mask_list = open('others/utt_tian_all.txt','r',encoding="utf-8").readlines()
        for utt_mask_ in utt_mask_list:
            utt_mask_ = [int(utt_mask_id) for utt_mask_id in utt_mask_.strip().split(' ')]
            utt_mask.append(utt_mask_)
        # print(utt_mask)
        
        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            utt_start = 0
            for batch in data_iter:
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                # print(batch.src_str)
                # TODO: 根据前一步生成的utt mask 进行 batch src len维度的扩充
                utt_mask_cur = utt_mask[utt_start:utt_start+batch.batch_size]
                # batch 个 topic_num 个 max_src 长度
                b_utt_mask_cur_expand = []
                # print(utt_mask_cur)
                # 统计每一个utt的长度：
                # print(batch.src.size())
                src_len = batch.src.size(1)
                # print(batch.diag)
                for i in range(batch.batch_size):
                    # with open('diag.txt','a+',encoding="utf-8") as ft:
                    #     # print(batch.diag[i])
                    #     ft.write(batch.diag[i][0])
                    #     ft.write('\n')

                    utt_mask_cur_expand = []
                    # 统计每个utt长度：
                    utt_lens = batch.src_lens[i]
                    # 按照2来分段：(修改为按照tag分段)
                    # utt_mask_segs = []
                    utt_mask_segs = [[],[],[],[],[],[]]
                    utt_mask_seg = []
                    # for j in utt_mask_cur[i]:
                    #     # if j != 2:
                    #     if j not in [2,3,4,5,6,7]:
                    #         utt_mask_seg.append(j-8)
                    #     else:
                    #         utt_mask_segs.append(utt_mask_seg)
                    #         utt_mask_seg = []
                    for j in utt_mask_cur[i]:
                        # if j != 2:
                        if j not in [2,3,4,5,6,7]:
                            utt_mask_seg.append(j-8)
                        else:
                            utt_mask_segs[j-2].extend(utt_mask_seg)
                            # utt_mask_segs.append(utt_mask_seg)
                            utt_mask_seg = []
                    # print(utt_mask_segs)
                    # 按照utt mask seg进行扩充：
                    for utt_seg in utt_mask_segs:
                        utt_mask_cur_ = []
                        for idx, utt_len in enumerate(utt_lens):
                            if idx in utt_seg:
                                utt_mask_cur_.extend([1]*utt_len)
                            else:
                                utt_mask_cur_.extend([0]*utt_len)
                        # utt_mask_cur_(src长度不够的要补上0)
                        if len(utt_mask_cur_) < src_len:
                            utt_mask_cur_.extend([0]*(src_len-len(utt_mask_cur_)))
                        # print(len(utt_mask_cur_))
                        utt_mask_cur_expand.append(utt_mask_cur_)
                        # print(len(utt_mask_cur_expand))

                    b_utt_mask_cur_expand.append(utt_mask_cur_expand)

                    # print(utt_len)
                # print(b_utt_mask_cur_expand)
                utt_start += batch.batch_size

                # decoder inference part :
                
                batch_data = self.translate_batch(batch, b_utt_mask_cur_expand)
                translations = self.from_batch(batch_data)

                # together 的情况 将user agent分开计算看一下：
                for trans in translations:
                    # pred, gold_u, gold_a, src = trans
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused0]', '').replace('[PAD]', '')\
                                    .replace('[unused1]', '').replace(r' +', ' ').replace(' [unused3] ', '<q>').replace('[unused3]', '')\
                                     .replace(' [unused4] ', '<q>').replace('[unused4]', '').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                    # pred_both = pred.replace('[unused0]', '').replace('[PAD]', '')\
                    #                 .replace('[unused1]', '').replace(r' +', ' ')\
                    #                  .replace(' [unused4] ', '<q>').replace('[unused4]', '').strip()
                    # pred_user = '<q>'.join([pred_each.replace(' [unused3] ', '<q>').replace('[unused3]', '').strip().split('<q>')[0] for pred_each in pred_both.split('<q>')])
                    # pred_agent = '<q>'.join([pred_each.replace(' [unused3] ', '<q>').replace('[unused3]', '').strip().split('<q>')[-1] for pred_each in pred_both.split('<q>')])

                    # gold_u, gold_a= gold_u.strip(), gold_a.strip()
                    gold_str = gold.strip()
                    if(self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str+ '<q>'+sent.strip()
                            can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                            # if(can_gap>=gap):
                            if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str



                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    # self.gold_u_path_file.write(gold_u + '\n')
                    # self.gold_a_path_file.write(gold_a + '\n')
                    # self.can_u_path_file.write(pred_user + '\n')
                    # self.can_a_path_file.write(pred_agent + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                # self.can_out_file.flush()
                # self.gold_u_path_file.flush()
                # self.gold_a_path_file.flush()
                # self.can_u_path_file.flush()
                # self.can_a_path_file.flush()
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        # self.can_out_file.close()
        # self.gold_u_path_file.close()
        # self.gold_a_path_file.close()
        # self.can_u_path_file.close()
        # self.can_a_path_file.close()
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        # if (step != -1):
        #     rouges = self._report_rouge(gold_path, can_path)
        #     self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        #     if self.tensorboard_writer is not None:
        #         self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
        cal_rouge_path(can_path, gold_path)
        
        # cal_rouge_path(can_u_path, gold_u_path)
        # cal_rouge_path(can_a_path, gold_a_path)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, b_utt_mask_cur_expand=None, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch, b_utt_mask_cur_expand,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch, b_utt_mask_cur_expand, 
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        # print(batch_size)
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        clss = batch.clss
        mask_cls = batch.mask_cls
        role_mask = batch.role_mask
        # print(role_mask.size())


        src_features = self.model.bert(src, segs, mask_src)

        dec_states = self.model.tog_decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        # b*beam_size, utt_num, hidden 
        src_features = tile(src_features, beam_size, dim=0)
        # b, beam顺序
        src = tile(src, beam_size, dim=0)
        role_mask = tile(role_mask, beam_size, dim=0)
        # [0...b-1]
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        #[0,beam-1,...,batch_size * beam_size-1]
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        # 保存每一次预测的结果id(初始化为start token id)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)
        # 初始化utt_mask：
        utt_mask_final = []
        # batch * seg first content(src)
        for utt_mask_cur_expand in b_utt_mask_cur_expand:
            utt_mask_final.append(torch.tensor(utt_mask_cur_expand[0]).to(role_mask))
        first_utt_mask = torch.stack(utt_mask_final)
        # print(first_utt_mask)
        first_utt_mask = tile(first_utt_mask, beam_size, dim=0)
        # record 当前的更新topic的次数 b*beam
        update_st = torch.ones([batch_size * beam_size]).to(first_utt_mask.device)

        # Give full probability to the first beam on the first step.
        # 初始化后作为每一次inference输出的id分数 (b)
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        # 最终输出结果：
        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        # 进入按步骤的解码
        # 当 beam_size=1时，下面的向量第一维均为batch_size
        for step in range(max_length):
            # 每次取最后的那个token作为输入：
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward. (b,1)
            decoder_input = decoder_input.transpose(0,1)

            # 调用decoder(此处对于topic sequence需要进行处理)
            # 判断转换为src utt hidden state 还是 tag embedding :
            # print(src_features.size())
            assert role_mask.size(0) == src_features.size(0)

            # TODO: 加入已经生成好的utt mask(判断然后更换mask序列)
            # 如果mask个数多于生成段落数，舍弃； 少于则默认为src pad mask(即全关注)

            # dec_out, dec_states, _, _, last_layer_score = self.model.tog_decoder(decoder_input, src_features, dec_states, role_mask, src_tgt_mask_tog=first_utt_mask,\
            #                                          step=step)
            dec_out, dec_states, last_layer_score = self.model.tog_decoder(decoder_input, src_features, dec_states, role_mask, src_tgt_mask_tog=first_utt_mask,\
                                                        step=step)

            # Generator forward.
            # png net:
            output_cat = torch.cat((torch.bmm(last_layer_score[0],src_features), dec_out),-1)
            # b*beam, vocab
            log_probs = self.generator.forward([dec_out,output_cat], last_layer_score, src)
            # log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))


            # 根据生成的针对vocab的概率选取词id
            vocab_size = log_probs.size(-1)

            # 在未到最小长度前禁止分给eos token值
            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.(b,1)
            # b*beam, vocab += b*beam_size,1
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            # 长度惩罚：
            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            # Trigram Blocking strategy
            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                # if(cur_len>3):
                if(cur_len>5):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##','').split()
                        # if(len(words)<=3):
                        #     continue
                        if(len(words)<=5):
                            continue
                        # trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigrams = [(words[i-2],words[i-1],words[i],words[i+1], words[i+2]) for i in range(2,len(words)-2)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            # b, beam_size * vocab_size
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            # b, beam_size
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            # print(topk_scores.size())

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids. (b, beam_size)
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)
            # print(topk_beam_index)
            # print(topk_ids)

            # Map beam_index to batch_index in the flat representation.
            # b, beam 
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            # alive_seq： b*beam, cur_len
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices.to(torch.long)),
                 topk_ids.view(-1, 1)], -1)

            #TODO: 增加更换utt mask的步骤：
            is_changed = topk_ids.eq(self.change_token).view(-1) # b,beam
            if is_changed.any():
                # 更新：b*beam
                for i in range(is_changed.size(0)):
                    # b*beam,src
                    if is_changed[i]: 
                        # 更新
                        update_st[i] += 1
                        # 判断是否还能继续更新，若不能则选择全部attn:
                        if int(update_st[i]) > len(b_utt_mask_cur_expand[i//beam_size]):
                            first_utt_mask[i] = torch.ones(first_utt_mask[i].size()).to(first_utt_mask[i].device)
                        else:
                            first_utt_mask[i] = torch.tensor(b_utt_mask_cur_expand[i//beam_size][int(update_st[i])-1]).to(first_utt_mask[i].device)

            is_finished = topk_ids.eq(self.end_token) # b,beam
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)  # 每个sample最大可能性的  #batch
            # Save finished hypotheses.
            if is_finished.any():
                # b, beam, cur_len
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    # batch_offset: batch 长度 [0...b-1]
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                # batch (non_finished: 未结束的batch id)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                # utt mask 也要跟着更新：
                b_utt_mask_cur_expand_ = []
                for i in list(non_finished.cpu().numpy()):
                    b_utt_mask_cur_expand_.append(b_utt_mask_cur_expand[i])
                b_utt_mask_cur_expand = b_utt_mask_cur_expand_
            # Reorder states.
            select_indices = batch_index.view(-1)
            first_utt_mask = first_utt_mask.index_select(0, select_indices.to(torch.long))
            src_features = src_features.index_select(0, select_indices.to(torch.long))
            src = src.index_select(0, select_indices.to(torch.long))
            role_mask = role_mask.index_select(0, select_indices.to(torch.long))
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices.to(torch.long)))

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
