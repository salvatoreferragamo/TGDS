#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None, label2vid=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger,label2vid=label2vid)
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
                 dump_beam="",
                 label2vid=None):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        # self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        # self.start_token = symbols['BOS']
        self.start_token = 0
        self.end_token = symbols['EOS']
        # self.src_start_index = 3
        # self.mapping = torch.LongTensor([99, 1, 2])
        self.label_ids = list(label2vid.values())
        self.mapping = torch.LongTensor([99, 1]+self.label_ids)
        self.src_start_index = len(self.mapping)

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

        # preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src
        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.topic_target[:,1:], batch.src

        # print(preds)
        # print('@@@@@@@@@@@')
        # print(tgt_str)

        translations = []
        # for b in range(batch_size):
        #     pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
        #     pred_sents = ' '.join(pred_sents).replace(' ##','')
        #     gold_sent = ' '.join(tgt_str[b].split())
        #     # translation = Translation(fname[b],src[:, b] if src is not None else None,
        #     #                           src_raw, pred_sents,
        #     #                           attn[b], pred_score[b], gold_sent,
        #     #                           gold_score[b])
        #     # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
        #     raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
        #     raw_src = ' '.join(raw_src)
        #     translation = (pred_sents, gold_sent, raw_src)
        #     # translation = (pred_sents[0], gold_sent)
        #     translations.append(translation)
        # print(tgt_str.size())
        # print(preds)
        # print(tgt_str)
        # print(batch.src_str)
        for b in range(batch_size):
            # 清洗：去掉重复的数字/ 最后一个2与1之间要清洗掉多余的utt id
            preds_list = []
            for idx,pred in enumerate(preds[b][0].cpu().numpy()):
                if idx !=0:
                    if pred == preds[b][0].cpu().numpy()[idx-1]:
                        continue
                    else:
                        preds_list.append(pred)
                else:
                    if pred in [2,3,4,5,6,7]:
                        continue
                    else:
                        preds_list.append(pred)
            # 分段：（相同的合并）
            # seg_id = [idx for idx,preds_id in enumerate(preds_list) if preds_id in [2,3,4,5,6,7]]
            # preds_list = preds_list[:seg_id[-1]+1]+[1]
            utt_list = [[],[],[],[],[],[]]
            id_list = []
            cur_list = []
            for idx,preds_id in enumerate(preds_list):
                if preds_id == 1:
                    break 
                else:
                    if preds_id in [2,3,4,5,6,7]:
                        utt_list[preds_id-2].extend(cur_list)
                        cur_list = []
                    else:
                        cur_list.append(preds_id)

            # 整理结果：
            preds_list = []
            for idx, u_l in enumerate(utt_list):
                if len(u_l) == 0:
                    continue
                else:
                    preds_list.extend(list(set(u_l)))
                    preds_list.append(idx+2)
            preds_list.append(1)

            pred_sents = [str(i) for i in preds_list]
            # 去除0
            # print(tgt_str[b])
            eos_id = [id for id,str in enumerate(tgt_str[b].cpu().numpy()) if str==1]
            gold_sent = list(tgt_str[b].cpu().numpy())[:eos_id[0]] + [1]
            gold_sent = [str(i) for i in gold_sent]
            # print(pred_sents, gold_sent)

            translation = (pred_sents, gold_sent)

            translations.append(translation)

        return translations

    # 用来构建topic sequence的生成器：
    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_src_path = self.args.result_path + '.%d.raw_src' % step
        # self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold = trans
                    pred_str = ' '.join(list(pred))
                    gold_str = ' '.join(list(gold))
                    # pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                    # gold_str = gold.strip()
                    # if(self.args.recall_eval):
                    #     _pred_str = ''
                    #     gap = 1e3
                    #     for sent in pred_str.split('<q>'):
                    #         can_pred_str = _pred_str+ '<q>'+sent.strip()
                    #         can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                    #         # if(can_gap>=gap):
                    #         if(len(can_pred_str.split())>=len(gold_str.split())+10):
                    #             pred_str = _pred_str
                    #             break
                    #         else:
                    #             gap = can_gap
                    #             _pred_str = can_pred_str



                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()

        from cal_rouge import cal_rouge_path
        cal_rouge_path(can_path, gold_path)
        # if (step != -1):
        #     rouges = self._report_rouge(gold_path, can_path)
        #     self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        #     if self.tensorboard_writer is not None:
        #         self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
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
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        clss = batch.clss
        mask_cls = batch.mask_cls
        avg_feature = True

        top_vec = self.model.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        src_features = self.model.ext_layer(sents_vec, mask_cls)
        # decoder state 开始起作用 ：调用state._init_cache
        dec_states = self.model.topic_decoder.init_decoder_state(sents_vec, src_features, with_cache=True)
        device = top_vec.device

        # src_features = self.model.bert(src, segs, mask_src)
        # dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        # device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        # b*beam_size, utt_num, hidden 
        src_features = tile(src_features, beam_size, dim=0)
        sents_vec = tile(sents_vec, beam_size, dim=0)
        mask_cls = tile(mask_cls, beam_size, dim=0)
        # [0...b-1]
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
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
            # tgt_pad_mask: (b,1)
            if step == 0:
                tgt_pad_masks = ~decoder_input.data.eq(1)
            else:
                tgt_pad_masks = ~decoder_input.data.eq(0)

            # 调用decoder(此处对于topic sequence需要进行处理)
            # TODO:判断转换为src utt hidden state 还是 tag embedding :
            mapping_token_mask = decoder_input.lt(self.src_start_index)  
            mapped_tokens = decoder_input.masked_fill(decoder_input.ge(self.src_start_index), 0)
            #b,1
            tag_mapped_tokens = self.mapping[mapped_tokens].to(mapped_tokens)\
            # src part
            src_tokens_index = decoder_input - self.src_start_index # bsz x tgt_num
            # b,1
            src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
            # 确保一下整体的范围未超出src的长度
            if step != 0:
                for i in range(src_tokens_index.size(0)):
                    assert int(src_tokens_index[i].max()) < len(sents_vec[i])
            # change to hidden states:
            utt_mapped_list = []
            for i in range(sents_vec.size(0)):
                utt_mapped_list.append(torch.index_select(sents_vec[i], 0, src_tokens_index[i]))
            # b, 1, h
            utt_mapped_vec = torch.stack(utt_mapped_list).to(sents_vec)
            # tag embedding : b, 1, emb_h
            tag_mapped_vec = self.model.topic_decoder.embeddings(tag_mapped_tokens)
            # 根据condition进行选取：
            decoder_input_list = []
            for i in range(mapping_token_mask.size(0)):
                condition = mapping_token_mask[i].unsqueeze(-1)
                # print(condition)
                # print(tag_mapped_vec[i], utt_mapped_vec[i])
                decoder_input_list.append(torch.where(condition, tag_mapped_vec[i], utt_mapped_vec[i]))
            # b,1,h
            decoder_input = torch.stack(decoder_input_list).to(utt_mapped_vec)

            # decoder part: (dec_out: b,1,h)
            dec_out, dec_states = self.model.topic_decoder(decoder_input, src_features, dec_states,memory_masks=~mask_cls,tgt_masks=tgt_pad_masks,
                                                     step=step)

            # TODO: Generator forward. 
            # log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            # b, tgt, vocab_size(tag+2+cur_utt_num)
            logits = dec_out.new_full((dec_out.size(0), dec_out.size(1), self.src_start_index+sents_vec.size(1)),
                                        fill_value=-1e24)

            # 首先计算的是 end 标签以及 Tag 标签：
            eos_scores = F.linear(dec_out, self.model.dropout_layer(self.model.topic_decoder.embeddings.weight[1:2]))  # bsz x max_len x 1
            pad_scores = F.linear(dec_out, self.model.dropout_layer(self.model.topic_decoder.embeddings.weight[0:1]))
            mapping_ids = torch.LongTensor(self.label_ids)
            # tag_scores = F.linear(dec_out, self.dropout_layer(self.topic_decoder.embeddings.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
            tag_scores = F.linear(dec_out, self.model.dropout_layer(self.model.topic_decoder.embeddings.weight[mapping_ids]))
            # tag_scores = F.linear(dec_out, self.model.dropout_layer(self.model.topic_decoder.embeddings.weight[2:3]))

            # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
            # b, utt_num, h
            if hasattr(self.model, 'encoder_mlp'):
                context_vec = self.model.encoder_mlp(src_features)

            # 先把feature合并一下 
            sents_vec = self.model.dropout_layer(sents_vec)
            if avg_feature: 
                context_vec = (context_vec + sents_vec)/2
            # 计算utt hidden 与 decoder hidden 相关得分
            # b, tgt_len, utt_len 
            word_scores = torch.einsum('blh,bnh->bln', dec_out, context_vec)  # bsz x max_len x max_word_len
            if not avg_feature:
                gen_scores = torch.einsum('blh,bnh->bln', dec_out, sents_vec)  # bsz x max_len x max_word_len
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

            #b, vocab_size
            log_probs = logits.squeeze(1)

            # 根据生成的针对vocab的概率选取词id
            vocab_size = log_probs.size(-1)

            # 在未到最小长度前禁止分给eos token值
            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.(b,1)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            # 长度惩罚：
            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.(b, vocab_size)
            curr_scores = log_probs / length_penalty

            # TODO:Trigram Blocking strategy(修改)
            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        # 按照2的位置进行分块：
                        if(2 not in words):
                            continue
                        seg_id= [-1] + [idx for idx,w in enumerate(words) if w == 2]
                        seg_list = [words[seg_id[id]+1:seg_id[id+1]] for id in range(len(seg_id)-1)] + [words[seg_id[-1]:]]
                        seggram = seg_list[-1]
                        for seg_i in seg_list[:-1]:
                            if len(set(seg_i) & set(seggram)) != 0:
                                fail = True
                        if fail:
                            curr_scores[i] = -10e20
                        # words = [self.vocab.ids_to_tokens[w] for w in words]
                        # words = ' '.join(words).replace(' ##','').split()
                        # if(len(words)<=3):
                        #     continue
                        # trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        # trigrams = [(words[i-1],words[i]) for i in range(1,len(words))]
                        # trigram = tuple(trigrams[-1])
                        # if trigram in trigrams[:-1]:
                        #     fail = True
                        # if fail:
                        #     curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices.to(torch.long)),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
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
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices.to(torch.long))
            sents_vec = sents_vec.index_select(0, select_indices.to(torch.long))
            mask_cls = mask_cls.index_select(0, select_indices.to(torch.long))
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
