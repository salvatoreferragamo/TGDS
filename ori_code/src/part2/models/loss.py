"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.reporter import Statistics


def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, last_layer_score=None):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        #shard_state = self._make_shard_state(batch, output)
        # _, batch_stats = self._compute_loss(batch, output, [batch.tgt_user[:, 1:], batch.tgt_agent[:, 1:]])
        _, batch_stats = self._compute_loss(batch, output, batch.tgt_all[:, 1:], last_layer_score=last_layer_score)

        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size,
                             normalization, args, scores=None, last_layer_score=None,
                             utt_scores=None, gold_utterances=None, mask_utterances=None):
    
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = [Statistics(), Statistics(), Statistics(), Statistics()]
        batch_stats = [Statistics(), Statistics()]
        # shard_state = self._make_shard_state(batch, output)
        # for shard in shards(shard_state, shard_size):
        # loss, stats = self._compute_loss(batch, output, [batch.tgt_user[:, 1:], batch.tgt_agent[:, 1:]], scores, utt_scores, gold_utterances, mask_utterances)
        loss, stats = self._compute_loss(batch, output, batch.tgt_all[:, 1:], scores, last_layer_score)
        # loss, stats = self._compute_loss(batch, output, batch.tgt_all[:, 1:], None, last_layer_score)
        # for i in range(2):
        #     loss[i] = loss[i].div(float(normalization[i]))
        #     batch_stats[i].update(stats[i])
        loss[0] = loss[0].div(float(normalization[0]))
        batch_stats[0].update(stats[0])
        
        # loss[-1] = loss[-1].div(float(normalization[-1]))
        # 召回率与准确率
        # batch_stats[-1].update(stats[-1])
        # batch_stats[-2].update(stats[-2])

        #print(loss[2], loss[3])
        # total_loss = args.role_weight * loss[0] + (1 - args.role_weight) * loss[1] + (loss[2] + loss[3]) * args.kl_weight + loss[-1] * args.l_weight
        # total_loss = args.role_weight * loss[0] + (1 - args.role_weight) * loss[1] + (loss[2] + loss[3]) * args.kl_weight 
        # total_loss = loss[0] + (loss[1] + loss[2]) * args.kl_weight 
        total_loss = loss[0]
        total_loss.backward()

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        # self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        self.label_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        # return {
        #     "output": output,
        #     "target": [batch.tgt_final[:,1:], batch.tgt_user[:, 1:], batch.tgt_agent[:, 1:]],
        # }

        return {
            "output": output[0],
            "target": batch.tgt_final[:, 1:],
        }

    def _compute_loss(self, batch, output, target, score=None, last_layer_score=None, utt_scores=None, gold_utterances=None, mask_utterances=None):
        '''
        score: user_scores, agent_scores, kl_mask_user, kl_mask_agent
        utt_scores : batch, target_l, source_l, 2
        gold_utterances[i] : batch, target_l, source_l
        '''
        losses, stats = [], []

        # NLL Loss
        # for i in range(2):
        #     bottled_output = self._bottle(output[i])
        #     scores = self.generator(bottled_output)
        #     gtruth = target[i].contiguous().view(-1)

        #     losses.append(self.criterion(scores, gtruth))

        #     stats.append(self._stats(losses[i].clone(), scores, gtruth))

        #TODO: together decoder part 
        # output[0], output[1]: [outputs,output_cat]
        # last_layer_score: user_score, agent_score (b,tgt,src)
        # bottled_output = self._bottle(output)
        # 调用generator
        # scores = self.generator(bottled_output)
        enc_batch_extend_vocab = batch.src
        scores = self.generator(output, last_layer_score, enc_batch_extend_vocab)
        gtruth = target.contiguous().view(-1)
        losses.append(self.criterion(scores, gtruth))
        stats.append(self._stats(losses[0].clone(), scores, gtruth))

        # KL loss
        #TODO: together decoder 需修改下面代码, 解决出现non值得问题
        if score is not None:
            # print(score[0][0][0].shape, score[2].shape)
            # exit()
            # concat different layer results(klloss部分)
            
            # (6 * bs, tgt_len_user, src_len)
            user_dec_user_att = torch.cat(score[0][0], dim=0).view(-1, score[2].shape[1], score[2].shape[2]) # (6 * bs, tgt_len_user, src_len)
            user_dec_agent_att = torch.cat(score[1][0], dim=0).view(-1, score[2].shape[1], score[2].shape[2])  # (12 * bs, tgt_len_user, src_len)
            agent_dec_user_att = torch.cat(score[0][1], dim=0).view(-1, score[3].shape[1], score[3].shape[2])  # (12 * bs, tgt_len_agent, src_len)
            agent_dec_agent_att = torch.cat(score[1][1], dim=0).view(-1, score[3].shape[1], score[3].shape[2])  # (12 * bs, tgt_len_agent, src_len)
            # do softmax and multiply mask

            score[2] = score[2].unsqueeze(0).expand(6, score[2].shape[0], score[2].shape[1], score[2].shape[2])\
                .reshape(-1, score[2].shape[1], score[2].shape[2])
            score[3] = score[3].unsqueeze(0).expand(6, score[3].shape[0], score[3].shape[1], score[3].shape[2]) \
                .reshape(-1, score[3].shape[1], score[3].shape[2])

            user_dec_user_att = F.softmax(user_dec_user_att, dim=-1) * score[2]  # (6 * bs, user_len, src_len)
            user_dec_agent_att = F.softmax(user_dec_agent_att, dim=-1) * score[2]  # (12 * bs, user_len, src_len)
            agent_dec_user_att = F.softmax(agent_dec_user_att, dim=-1) * score[3]  # (12 * bs, agent_len, src_len)
            agent_dec_agent_att = F.softmax(agent_dec_agent_att, dim=-1) * score[3]  # (12 * bs, agent_len, src_len)
            
            # average on tgt len dim
            user_dec_user_att = torch.sum(user_dec_user_att, dim=1) / torch.sum(score[2], dim=1)
            user_dec_agent_att = torch.sum(user_dec_agent_att, dim=1) / torch.sum(score[2], dim=1)
            agent_dec_user_att = torch.sum(agent_dec_user_att, dim=1) / torch.sum(score[3], dim=1)
            agent_dec_agent_att = torch.sum(agent_dec_agent_att, dim=1) / torch.sum(score[3], dim=1)

            user_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(agent_dec_user_att + 1e-12), user_dec_user_att + 1e-12)
            agent_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(user_dec_agent_att + 1e-12), agent_dec_agent_att + 1e-12)
            
            losses += [user_loss, agent_loss]
        ###
        # if score is not None:
        #     # print(score[0][0][0].shape, score[2].shape)
        #     # exit()
        #     # concat different layer results(klloss部分)
        #     user_dec_user_att = torch.cat(score[0][0], dim=0).view(-1, score[2].shape[1], score[2].shape[2]) # (12 * bs, tgt_len_user, src_len)
        #     user_dec_agent_att = torch.cat(score[1][0], dim=0).view(-1, score[2].shape[1], score[2].shape[2])  # (12 * bs, tgt_len_user, src_len)
        #     agent_dec_user_att = torch.cat(score[0][1], dim=0).view(-1, score[3].shape[1], score[3].shape[2])  # (12 * bs, tgt_len_agent, src_len)
        #     agent_dec_agent_att = torch.cat(score[1][1], dim=0).view(-1, score[3].shape[1], score[3].shape[2])  # (12 * bs, tgt_len_agent, src_len)
        #     # do softmax and multiply mask
        #     score[2] = score[2].unsqueeze(0).expand(6, score[2].shape[0], score[2].shape[1], score[2].shape[2])\
        #         .reshape(-1, score[2].shape[1], score[2].shape[2])
        #     score[3] = score[3].unsqueeze(0).expand(6, score[3].shape[0], score[3].shape[1], score[3].shape[2]) \
        #         .reshape(-1, score[3].shape[1], score[3].shape[2])

        #     user_dec_user_att = F.softmax(user_dec_user_att, dim=-1) * score[2]  # (6 * bs, user_len, src_len)
        #     user_dec_agent_att = F.softmax(user_dec_agent_att, dim=-1) * score[2]  # (12 * bs, user_len, src_len)
        #     agent_dec_user_att = F.softmax(agent_dec_user_att, dim=-1) * score[3]  # (12 * bs, agent_len, src_len)
        #     agent_dec_agent_att = F.softmax(agent_dec_agent_att, dim=-1) * score[3]  # (12 * bs, agent_len, src_len)
        #     # average on tgt len dim
        #     user_dec_user_att = torch.sum(user_dec_user_att, dim=1) / torch.sum(score[2], dim=1)
        #     user_dec_agent_att = torch.sum(user_dec_agent_att, dim=1) / torch.sum(score[2], dim=1)
        #     agent_dec_user_att = torch.sum(agent_dec_user_att, dim=1) / torch.sum(score[3], dim=1)
        #     agent_dec_agent_att = torch.sum(agent_dec_agent_att, dim=1) / torch.sum(score[3], dim=1)
        #     # print(torch.sum(user_dec_user_att, dim=1))
        #     # print(torch.sum(user_dec_agent_att, dim=1))
        #     # print(torch.sum(agent_dec_user_att, dim=1))
        #     # print(torch.sum(agent_dec_agent_att, dim=1))
        #     user_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(agent_dec_user_att + 1e-12), user_dec_user_att + 1e-12)
        #     agent_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(user_dec_agent_att + 1e-12), agent_dec_agent_att + 1e-12)

        #     # user_loss = torch.sum(
        #     #     torch.nn.KLDivLoss(reduction='none')(F.log_softmax(agent_dec_user_att, 1), F.softmax(user_dec_user_att, 1)), 1)
        #     # agent_loss = torch.sum(
        #     #     torch.nn.KLDivLoss(reduction='none')(F.log_softmax(user_dec_agent_att, 1), F.softmax(agent_dec_agent_att, 1)), 1)
        #     # user_loss = torch.sum(user_loss.view(-1, score[2].shape[1]), dim=1) / torch.sum(score[2][:, :, 0].unsqueeze(0).expand(12, score[2].shape[0], score[2].shape[1]).reshape(-1, score[2].shape[1]))
        #     # agent_loss = torch.sum(agent_loss.view(-1, score[2].shape[1]), dim=1) / torch.sum(
        #     #     score[2][:, :, 0].unsqueeze(0).expand(12, score[2].shape[0], score[2].shape[1]).reshape(-1,
        #     #                                                                                          score[2].shape[1]))  # (12 * bs)
        #     # user_loss = torch.mean(user_loss)
        #     # agent_loss = torch.mean(agent_loss)
        #     losses += [user_loss, agent_loss]

        # user与agent合在一起：
        # utt_scores: b,tgt,src,2
        if utt_scores is not None:
            loss = self.label_criterion(utt_scores.view(-1, utt_scores.size(-1)), gold_utterances.view(-1))
            loss = (loss * mask_utterances.float().view(-1)).sum()
            losses.append(loss)  
            # 计算准确率和召回率：
            src, dim = utt_scores.size(-2), utt_scores.size(-1)
            # import torch.nn.functional as F
            # import numpy as np
            sent_scores = F.softmax(utt_scores.reshape(-1,src,dim),-1).cpu().data.numpy()
            # B*TGT,SRC
            selected_ids = np.argmax(sent_scores, 2)
            labels = gold_utterances.reshape(-1,src).cpu().data.numpy()
            mask_utterances = mask_utterances.reshape(-1,src).cpu().data.numpy()
            # batch
            gold = []
            pred = []
            for i in range(len(selected_ids)):
                _pred = []
                _correct = []
                # 不padding的真实长度
                for j in range(int(mask_utterances[i].sum())):
                    _pred.append(selected_ids[i][j])
                    _correct.append(labels[i][j])
                gold.append(_correct)
                pred.append(_pred)
            
            foundPredCnt, GoldCnt, correctChunkCnt = computePredict(gold, pred)

            stats.append(Statistics(loss.clone().item(),foundPredCnt,correctChunkCnt))  
            stats.append(Statistics(loss.clone().item(),GoldCnt,correctChunkCnt)) 
        # if utt_scores is not None:
        #     # INTENT Loss
        #     for i in range(2):
        #         loss = self.label_criterion(utt_scores[i].view(-1, utt_scores[i].size(-1)), gold_utterances[i].view(-1))
        #         loss = (loss * mask_utterances[i].float().view(-1)).sum()
        #         losses.append(loss / loss.numel())


        return losses, stats


def computePredict(correct_das, pred_das):
        '''
        correct_das, pred_das
        '''
        correctChunkCnt = 0
        foundPredCnt = 0
        GoldCnt = 0
        # print(pred_das)
        for correct_da, pred_da in zip(correct_das, pred_das):
            assert len(correct_da) == len(pred_da)
            # 找1的位置而不是直接比较一样:
            gold_index = []
            predict_index = []
            step = 0
            for c, p in zip(correct_da, pred_da):
                if c == 1:
                    gold_index.append(step)
                if p == 1:
                    predict_index.append(step)
                step+=1
            # 统计相同的位置以及各自的多少:
            foundPredCnt += len(predict_index)
            GoldCnt += len(gold_index)
            correctChunkCnt += len(set(gold_index) & set(predict_index))

        return foundPredCnt, GoldCnt, correctChunkCnt


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))
        print(non_none)

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))
        print(values)
        #exit()

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
