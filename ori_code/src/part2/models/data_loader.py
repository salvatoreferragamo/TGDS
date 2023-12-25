import bisect
import gc
import glob
import random

import torch

from others.logging import logger
from pathlib import Path



class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    def pad_2d(self, data, pad_id , width=-1):
        # tgt_num
        width_1d = max(max(len(d) for d in data), width)
        # src_num
        width_2d = max(len(d[0]) for d in data)
        pad_2d = [pad_id] * width_2d
        rtn_data = []
        for d in data:
            rtn_data_ = []
            for x in d:
                rtn_data_.append(x + [pad_id] * (width_2d - len(x)))
            while len(rtn_data_) < width_1d :
                rtn_data_.append(pad_2d)
            rtn_data.append(rtn_data_)    
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt_user = [x[1] for x in data]
            pre_tgt_agent = [x[2] for x in data]
            pre_segs = [x[3] for x in data]
            pre_clss = [x[4] for x in data]
            role_mask = [x[5] for x in data]
            intent_label = [x[6] for x in data]
            pre_tgt_user_seg_idxs = [x[7] for x in data]
            pre_tgt_agent_seg_idxs = [x[8] for x in data]
            pre_src_lens = [x[9] for x in data]
            pre_tgt_user_lens = [x[10] for x in data]
            pre_tgt_agent_lens = [x[11] for x in data]
            user_utterances = [x[12] for x in data]
            agent_utterances = [x[13] for x in data]
            user_src_tgt_mask_final = [x[14] for x in data]
            agent_src_tgt_mask_final = [x[15] for x in data]
            merge_utterances = [x[16] for x in data]
            pre_tgt_all= [x[17] for x in data]
            pre_tgt_fin_role= [x[18] for x in data]
            pre_src_tgt_mask_final_tog = [x[19] for x in data]

            tgt_user_len = max([len(seq) for seq in pre_tgt_user])
            tgt_agent_len = max([len(seq) for seq in pre_tgt_agent])
            tgt_max_len = max(tgt_user_len, tgt_agent_len)

            src = torch.tensor(self._pad(pre_src, 0))
            tgt_all = torch.tensor(self._pad(pre_tgt_all, 0))
            # 和 tgt_all共用mask
            tgt_fin_role= torch.tensor(self._pad(pre_tgt_fin_role, 0))
            tgt_user = torch.tensor(self._pad(pre_tgt_user, 0, tgt_max_len))
            tgt_agent = torch.tensor(self._pad(pre_tgt_agent, 0, tgt_max_len))
            role_mask = torch.tensor(self._pad(role_mask, 0))
            # B, tgt, src
            user_utterances = torch.tensor(self.pad_2d(user_utterances, -1))
            agent_utterances = torch.tensor(self.pad_2d(agent_utterances, -1))
            merge_utterances = torch.tensor(self.pad_2d(merge_utterances, -1))
            # 此处之前有错误
            mask_user_utterances = ~(user_utterances == -1)
            mask_agent_utterances = ~(agent_utterances == -1)
            mask_merge_utterances = ~(merge_utterances == -1)

            user_utterances[user_utterances == -1] = 0
            agent_utterances[agent_utterances == -1] = 0
            merge_utterances[merge_utterances == -1] = 0

            # B, tgt_len, src_len
            user_src_tgt_mask_final = torch.tensor(self.pad_2d(user_src_tgt_mask_final, 0, tgt_max_len))
            agent_src_tgt_mask_final = torch.tensor(self.pad_2d(agent_src_tgt_mask_final, 0, tgt_max_len))
            # tog mask 
            src_tgt_mask_final_tog = torch.tensor(self.pad_2d(pre_src_tgt_mask_final_tog, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = ~(src == 0)
            mask_tgt_all = ~(tgt_all== 0)
            mask_tgt_user = ~(tgt_user == 0)
            mask_tgt_agent = ~(tgt_agent == 0)


            clss = torch.tensor(self._pad(pre_clss, -1))
            # 每个seg的长度（为后面的inference做铺垫）
            src_lens = torch.tensor(self._pad(pre_src_lens, 0))
            mask_src_lens = ~(src_lens == 0)

            tgt_user_lens = torch.tensor(self._pad(pre_tgt_user_lens, 0))
            mask_tgt_user_lens = ~(tgt_user_lens == 0)
            tgt_agent_lens = torch.tensor(self._pad(pre_tgt_agent_lens, 0))
            mask_tgt_agent_lens = ~(tgt_agent_lens == 0)

            tgt_user_seg_idxs = torch.tensor(self._pad(pre_tgt_user_seg_idxs, -1))
            tgt_agent_seg_idxs = torch.tensor(self._pad(pre_tgt_agent_seg_idxs, -1))

            mask_tgt_user_seg_idxs = ~(tgt_user_seg_idxs == -1) 
            tgt_user_seg_idxs[tgt_user_seg_idxs == -1] = 0
            mask_tgt_agent_seg_idxs = ~(tgt_agent_seg_idxs == -1) 
            tgt_agent_seg_idxs[tgt_agent_seg_idxs == -1] = 0

            # intent
            # intent_label = torch.tensor(self._pad(intent_label, -1))
            intent_label = torch.tensor(self._pad(intent_label, 0))
            mask_cls = ~(clss == -1) # 也可以做intent的mask
            clss[clss == -1] = 0
            # intent_label[intent_label == -1] = 0

            setattr(self, 'intent_label', intent_label.to(device))
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt_all', tgt_all.to(device))
            setattr(self, 'tgt_fin_role', tgt_fin_role.to(device))
            setattr(self, 'tgt_user', tgt_user.to(device))
            setattr(self, 'tgt_agent', tgt_agent.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt_all', mask_tgt_all.to(device))
            setattr(self, 'mask_tgt_user', mask_tgt_user.to(device))
            setattr(self, 'mask_tgt_agent', mask_tgt_agent.to(device))
            setattr(self, 'role_mask', role_mask.to(device))

            ## ADD 为了求句向量
            setattr(self, 'tgt_user_seg_idxs', tgt_user_seg_idxs.to(device))
            setattr(self, 'mask_tgt_user_seg_idxs', mask_tgt_user_seg_idxs.to(device))
            setattr(self, 'tgt_agent_seg_idxs', tgt_agent_seg_idxs.to(device))
            setattr(self, 'mask_tgt_agent_seg_idxs', mask_tgt_agent_seg_idxs.to(device))

            # len
            setattr(self, 'src_lens', src_lens.to(device))
            setattr(self, 'mask_src_lens', mask_src_lens.to(device))
            setattr(self, 'tgt_user_lens', tgt_user_lens.to(device))
            setattr(self, 'mask_tgt_user_lens', mask_tgt_user_lens.to(device))
            setattr(self, 'tgt_agent_lens', tgt_agent_lens.to(device))
            setattr(self, 'mask_tgt_agent_lens', mask_tgt_agent_lens.to(device))

            # golden label
            setattr(self, 'user_utterances', user_utterances.to(device))
            setattr(self, 'agent_utterances', agent_utterances.to(device))
            setattr(self, 'mask_user_utterances', mask_user_utterances.to(device))
            setattr(self, 'mask_agent_utterances', mask_agent_utterances.to(device))

            #mask 
            setattr(self, 'user_src_tgt_mask_final', user_src_tgt_mask_final.to(device))
            setattr(self, 'agent_src_tgt_mask_final', agent_src_tgt_mask_final.to(device))
            setattr(self, 'src_tgt_mask_final_tog', src_tgt_mask_final_tog.to(device))

            setattr(self, 'merge_utterances', merge_utterances.to(device))
            setattr(self, 'mask_merge_utterances', mask_merge_utterances.to(device))

            # 可视化:
            # tgt_fin_len = [x[20] for x in data]
            # tgt_fin_len = torch.tensor(self._pad(tgt_fin_len, 0))
            # setattr(self, 'tgt_fin_len', tgt_fin_len.to(device))

            if is_test:
                # 不同数据的长度不一样的attr被保留了起来
                src_str = [x[-3] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str_user = [x[-2] for x in data]
                setattr(self, 'tgt_str_user', tgt_str_user)
                tgt_str_agent = [x[-1] for x in data]
                setattr(self, 'tgt_str_agent', tgt_str_agent)
                tgt_str_fin = [x[-4] for x in data]
                setattr(self, 'tgt_str_fin', tgt_str_fin)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "val", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        # 测试用
        # dataset = torch.load(pt_file)[:10]
        dataset = torch.load(pt_file)
        # print(dataset[:10])
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    ptps = sorted((Path(args.bert_data_path)/corpus_type).iterdir())
    #ptps = (Path(args.bert_data_path) / corpus_type).iterdir()

    
    if (shuffle):
        random.shuffle(ptps)

    for pt in ptps:
        yield _lazy_dataset_loader(pt, corpus_type)
    

def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[2]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    #print(count, src_elements)
    # if (count > 6):
    #     return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[0], new[6]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
            
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        
        return xs

    # data transform
    def preprocess(self, ex, is_test):
        # 输入序列最大长度截断
        src = ex['src'][:self.args.max_src_len]
        tgt_user = ex['tgt_user'][:self.args.max_tgt_len]
        tgt_agent = ex['tgt_agent'][:self.args.max_tgt_len]
        tgt_all = ex['tgt_all'][:self.args.max_tgt_len_all]
        tgt_fin_role = ex['tgt_fin_role'][:self.args.max_tgt_len_all]
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt_user = ex['tgt_txt_user']
        tgt_txt_agent = ex['tgt_txt_agent']
        tgt_fin_txts = ex['tgt_fin_txts']
        role_mask = ex['role_mask']
        # 加一个意图识别的模块
        intent_label = ex['intent_label']
        ##### 后加的部分
        # seg part 
        tgt_user_seg_idxs = ex['tgt_user_seg_idxs']
        tgt_agent_seg_idxs = ex['tgt_agent_seg_idxs']
        # len part
        src_lens = ex['src_lens']
        tgt_user_lens = ex['tgt_user_lens']
        tgt_agent_lens = ex['tgt_agent_lens']
        # seg label
        user_utterances = ex['user_utterances']
        agent_utterances = ex['agent_utterances']
        merge_utterances = ex['merge_utterances']

        # mask src_tgt
        user_src_tgt_mask_final = [src_sent[:self.args.max_pos] for src_sent in ex['user_src_tgt_mask_final']][:self.args.max_tgt_len]
        agent_src_tgt_mask_final = [src_sent[:self.args.max_pos] for src_sent in ex['agent_src_tgt_mask_final']][:self.args.max_tgt_len]
        # user&agent一起的一个mask:
        src_tgt_mask_final_tog = [src_sent[:self.args.max_pos] for src_sent in ex['src_tgt_mask_final_tog']][:self.args.max_tgt_len_all]

        # sep id
        end_id = [src[-1]]
        # transformer接受512为max
        src = src[:-1][:self.args.max_pos - 1] + end_id
        role_mask = role_mask[:-1][:self.args.max_pos - 1] + [role_mask[-1]]
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        clss = clss[:max_sent_id]
        # utt_len 截取：

        # 过滤掉超出长度的段落
        user_utterances = [user_utterance[:max_sent_id] for user_utterance in user_utterances]
        agent_utterances = [agent_utterance[:max_sent_id] for agent_utterance in agent_utterances]
        merge_utterances = [merge_utterance[:max_sent_id] for merge_utterance in merge_utterances]
        # 此处要修改（每一段的长度）
        if sum(src_lens[:max_sent_id]) >  self.args.max_pos:
            src_lens = src_lens[:max_sent_id-1] + [self.args.max_pos-sum(src_lens[:max_sent_id-1])]
        else:
            src_lens = src_lens[:max_sent_id] 

        intent_label = intent_label[:max_sent_id]
        # 针对tgt的截断（有几个<q>就是几段）
        max_tgt_user_id = bisect.bisect_left(tgt_user_seg_idxs, self.args.max_tgt_len)
        max_tgt_agent_id = bisect.bisect_left(tgt_agent_seg_idxs, self.args.max_tgt_len)
        # start <q> 和 （end的位置记录）不要：
        # 保证user和agent 的seg数量相同(测试)
        same_seg = min(max_tgt_user_id,max_tgt_agent_id)
        max_tgt_user_id, max_tgt_agent_id = same_seg,same_seg

        tgt_user_seg_idxs = tgt_user_seg_idxs[:max_tgt_user_id]
        tgt_agent_seg_idxs = tgt_agent_seg_idxs[:max_tgt_agent_id]
        # 
        tgt_user_lens = tgt_user_lens[:max_tgt_user_id]
        tgt_agent_lens = tgt_agent_lens[:max_tgt_agent_id]
        # label 
        user_utterances = user_utterances[:len(tgt_user_lens)]
        agent_utterances = agent_utterances[:len(tgt_agent_lens)]
        # add
        merge_utterances = merge_utterances[:len(tgt_agent_lens)]
        #可视化:
        tgt_fin_len = ex['tgt_fin_len']

        # if(is_test):
        #     return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        # else:
        #     return src, tgt, segs, clss, src_sent_labels

        if (is_test):
            return src, tgt_user, tgt_agent,\
                   segs, clss, role_mask, intent_label, tgt_user_seg_idxs, tgt_agent_seg_idxs,\
                   src_lens, tgt_user_lens, tgt_agent_lens,\
                   user_utterances, agent_utterances,\
                   user_src_tgt_mask_final, agent_src_tgt_mask_final, merge_utterances,\
                   tgt_all, tgt_fin_role, src_tgt_mask_final_tog, tgt_fin_txts,\
                   src_txt, tgt_txt_user, tgt_txt_agent, 
        else:
            return src, tgt_user, tgt_agent,\
                   segs, clss, role_mask, intent_label, tgt_user_seg_idxs, tgt_agent_seg_idxs,\
                   src_lens, tgt_user_lens, tgt_agent_lens,\
                   user_utterances, agent_utterances,\
                   user_src_tgt_mask_final, agent_src_tgt_mask_final, merge_utterances, tgt_all, tgt_fin_role, src_tgt_mask_final_tog, tgt_fin_len

    # tianchi:
    # def preprocess(self, ex, is_test):
    #     # 输入序列最大长度截断
    #     src = ex['src'][:self.args.max_src_len]
    #     tgt_all = ex['tgt_all'][:self.args.max_tgt_len_all]
    #     tgt_fin_role = ex['tgt_fin_role'][:self.args.max_tgt_len_all]
    #     segs = ex['segs']
    #     if(not self.args.use_interval):
    #         segs=[0]*len(segs)
    #     clss = ex['clss']
    #     src_txt = ex['src_txt']
    #     tgt_txt_user = ex['tgt_txt_user']
    #     tgt_txt_agent = ex['tgt_txt_agent']
    #     tgt_fin_txts = ex['tgt_fin_txts']
    #     role_mask = ex['role_mask']
    #     # 加一个意图识别的模块
    #     intent_label = ex['intent_label']
    #     ##### 后加的部分
    #     # seg part 
    #     tgt_user_seg_idxs = ex['tgt_user_seg_idxs']
    #     tgt_agent_seg_idxs = ex['tgt_agent_seg_idxs']
    #     # len part
    #     src_lens = ex['src_lens']
    #     tgt_user_lens = ex['tgt_user_lens']
    #     tgt_agent_lens = ex['tgt_agent_lens']
    #     # seg label
    #     user_utterances = ex['user_utterances']
    #     agent_utterances = ex['agent_utterances']
    #     merge_utterances = ex['merge_utterances']

    #     # mask src_tgt
    #     user_src_tgt_mask_final = [src_sent[:self.args.max_pos] for src_sent in ex['user_src_tgt_mask_final']][:self.args.max_tgt_len]
    #     agent_src_tgt_mask_final = [src_sent[:self.args.max_pos] for src_sent in ex['agent_src_tgt_mask_final']][:self.args.max_tgt_len]
    #     # user&agent一起的一个mask:
    #     src_tgt_mask_final_tog = [src_sent[:self.args.max_pos] for src_sent in ex['src_tgt_mask_final_tog']][:self.args.max_tgt_len_all]

    #     # sep id
    #     end_id = [src[-1]]
    #     # transformer接受512为max
    #     src = src[:-1][:self.args.max_pos - 1] + end_id
    #     role_mask = role_mask[:-1][:self.args.max_pos - 1] + [role_mask[-1]]
    #     segs = segs[:self.args.max_pos]
    #     max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
    #     clss = clss[:max_sent_id]
    #     # utt_len 截取：

    #     # 过滤掉超出长度的段落
    #     user_utterances = [user_utterance[:max_sent_id] for user_utterance in user_utterances]
    #     agent_utterances = [agent_utterance[:max_sent_id] for agent_utterance in agent_utterances]
    #     merge_utterances = [merge_utterance[:max_sent_id] for merge_utterance in merge_utterances]
    #     # 此处要修改（每一段的长度）
    #     if sum(src_lens[:max_sent_id]) >  self.args.max_pos:
    #         src_lens = src_lens[:max_sent_id-1] + [self.args.max_pos-sum(src_lens[:max_sent_id-1])]
    #     else:
    #         src_lens = src_lens[:max_sent_id] 

    #     intent_label = intent_label[:max_sent_id]
    #     # 针对tgt的截断（有几个<q>就是几段）
    #     max_tgt_user_id = bisect.bisect_left(tgt_user_seg_idxs, self.args.max_tgt_len)
    #     max_tgt_agent_id = bisect.bisect_left(tgt_agent_seg_idxs, self.args.max_tgt_len)
    #     # start <q> 和 （end的位置记录）不要：
    #     # 保证user和agent 的seg数量相同(测试)
    #     same_seg = min(max_tgt_user_id,max_tgt_agent_id)
    #     max_tgt_user_id, max_tgt_agent_id = same_seg,same_seg

    #     tgt_user_seg_idxs = tgt_user_seg_idxs[:max_tgt_user_id]
    #     tgt_agent_seg_idxs = tgt_agent_seg_idxs[:max_tgt_agent_id]
    #     # 
    #     tgt_user_lens = tgt_user_lens[:max_tgt_user_id]
    #     tgt_agent_lens = tgt_agent_lens[:max_tgt_agent_id]
    #     # label 
    #     user_utterances = user_utterances[:len(tgt_user_lens)]
    #     agent_utterances = agent_utterances[:len(tgt_agent_lens)]
    #     # add
    #     merge_utterances = merge_utterances[:len(tgt_agent_lens)]
    #     #

    #     # if(is_test):
    #     #     return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
    #     # else:
    #     #     return src, tgt, segs, clss, src_sent_labels

    #     if (is_test):
    #         return src, tgt_user, tgt_agent,\
    #                segs, clss, role_mask, intent_label, tgt_user_seg_idxs, tgt_agent_seg_idxs,\
    #                src_lens, tgt_user_lens, tgt_agent_lens,\
    #                user_utterances, agent_utterances,\
    #                user_src_tgt_mask_final, agent_src_tgt_mask_final, merge_utterances,\
    #                tgt_all, tgt_fin_role, src_tgt_mask_final_tog, tgt_fin_txts,\
    #                src_txt, tgt_txt_user, tgt_txt_agent, 
    #     else:
    #         return src, tgt_user, tgt_agent,\
    #                segs, clss, role_mask, intent_label, tgt_user_seg_idxs, tgt_agent_seg_idxs,\
    #                src_lens, tgt_user_lens, tgt_agent_lens,\
    #                user_utterances, agent_utterances,\
    #                user_src_tgt_mask_final, agent_src_tgt_mask_final, merge_utterances, tgt_all, tgt_fin_role, src_tgt_mask_final_tog

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far >= batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            # elif size_so_far > batch_size:
            #     yield minibatch[:-1]
            #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            # elif size_so_far > batch_size:
            #     yield minibatch[:-1]
            #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        # 进一步控制batch
        for buffer in self.batch_buffer(data, self.batch_size * 200):
            
            # 按顺序排序
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[3]))

            p_batch = self.batch(p_batch, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            # elif size_so_far > batch_size:
            #     yield minibatch[:-1]
            #     minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 200):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[0]))
                #p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
