import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import collections
from collections import Counter
from os.path import join as pjoin
from pathlib import Path

import torch
from multiprocessing import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

import jieba
from itertools import chain
import re


nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]

def cut_paragraph(paragraph):
  splited = [] 
  separator = list('。；？！.;!?')
  sen = []
  is_divisible = True
  for char in paragraph:
    sen.append(char)
    if char == '“':
      is_divisible = False
    if char == '”':
      is_divisible = True
      
    if char in separator and is_divisible:
      splited.append(''.join(sen))
      sen = []
    
  if len(sen) != 0:
    splited.append(''.join(sen))

  return splited


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p):
    with p.open('r', encoding = 'utf-8') as f:
        json_data = json.load(f)
        article = json_data["article"]
        abstract = json_data["abstract"]
        source = [[tk.lower() for tk in sen.strip().split()] for sen in article]
        tgt = [[tk.lower() for tk in sen.strip().split()] for sen in abstract]

        source = [clean(' '.join(sent)).split() for sent in source]
        tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, pretrained_path, args, tag_mapping):
        self.args = args
        # 要新增加Tag special token 防止相关的tag label在编码时被bpe掉
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True, mapping=tag_mapping)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.eou = '[unused98]'
        self.tgt_bos = '[unused99]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        # 对于seg tgt的识别
        self.tgt_bos_vid = self.tokenizer.vocab[self.tgt_bos]
        self.tgt_eos_vid = self.tokenizer.vocab[self.tgt_eos]
        self.tgt_sent_split_vid = self.tokenizer.vocab[self.tgt_sent_split]


    def preprocess(self, data, is_print, use_bert_basic_tokenizer, is_test=False):

        if ((not is_test) and len(data['src']) == 0):
            return None

        original_src_txt = [' '.join(s) for s in data['src']]
        # 单句不够长度则去除，超过则截断
        idxs = [i for i, s in enumerate(data['src']) if (len(s) > self.args.min_src_ntokens_per_sent)]
        src = [data['src'][i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        # 超过最多句子则段落截断
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent) for sent in src]
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None
        role_label = data['role_label'][:self.args.max_src_nsents]
        intent_label = data['intent_label'][:self.args.max_src_nsents]
        # 对齐问题：
        # user_utterances 和 agent 这两个为了记录类别信息
        user_utterances = [user_utterances[:self.args.max_src_nsents] for user_utterances in data['user_utterances']]
        agent_utterances = [agent_utterances[:self.args.max_src_nsents] for agent_utterances in data['agent_utterances']]
        merge_utterances = [merge_utterances[:self.args.max_src_nsents] for merge_utterances in data['merge_utterances']]
        # tokenizer处理
        src_subtokens = [self.tokenizer.tokenize(text, use_bert_basic_tokenizer=use_bert_basic_tokenizer) for text in
                         src_txt]
        # role context 算进去了 cls和sep部分
        role_masks = [[role_label[i]] * (len(src_subtokens[i]) + 2) for i in range(len(role_label))]
        role_masks = [s for sent in role_masks for s in sent]
        # user_utterances 和 agent_utterances 要扩充为 src_tgt_mask
        # utter_num -> src_len ; seg_num -> tgt_len
        user_src_tgt_mask = []
        for user_utterance in merge_utterances:
            user_src_tgt_mask_ = []
            for i in range(len(user_utterance)):
                for k in range(len(src_subtokens[i]) + 2):
                    user_src_tgt_mask_.append(user_utterance[i])

            user_src_tgt_mask.append(user_src_tgt_mask_)
        agent_src_tgt_mask = []
        for agent_utterance in merge_utterances:
            agent_src_tgt_mask_ = []
            for i in range(len(agent_utterance)):
                for k in range(len(src_subtokens[i]) + 2):
                    agent_src_tgt_mask_.append(agent_utterance[i])
                    
            agent_src_tgt_mask.append(agent_src_tgt_mask_)


        src_subtoken_idxs = []
        src_subtoken_idxs_ = []
        # sep_src_subtoken_idxs = []
        sep_src_subtoken_idxs_ = []
        for tokens in src_subtokens:
            src_subtoken_idxs += [self.cls_token] + tokens + [self.sep_token]
            sep_src_subtoken_idxs_.append([self.cls_token] + tokens + [self.sep_token])

        src_subtoken_idxs_ = src_subtoken_idxs
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtoken_idxs)
        # utterance_num个utterance_ids
        sep_src_subtoken_idxs = [self.tokenizer.convert_tokens_to_ids(l) for l in sep_src_subtoken_idxs_]
        # segments_id
        sep_seg_id = []
        for i in range(len(sep_src_subtoken_idxs)):
            sep_seg_id.append([0]*len(sep_src_subtoken_idxs[i]))
        
        assert len(role_label) == len(sep_src_subtoken_idxs) == len(sep_seg_id)

        if len(src_subtoken_idxs) != len(role_masks):
            print(len(src_subtoken_idxs))
            print(len(role_masks))
            assert True
        # src_subtoken_idxs : list[id]
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        src_txt = [original_src_txt[i] for i in idxs]

        tgt_idxs = [[], []]
        tgt_txts = [[], []]
        tgt_subtokens = [[],[]]
        tgts = [data['tgt_user'], data['tgt_agent']]
        tgt_seg_idxs = [[], []]
        tgt_seg_idxs_ = [[], []]
        tgt_lens = [[], []]
        agent_id = data['agent_id']

        # 分为user和agent
        for i in range(2):
            tgt_subtokens_str = '[unused99] ' + ' [unused2] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
                 tt in tgts[i]]) + ' [unused1]'
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
            tgt_subtokens[i] = tgt_subtoken
            if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
                return None

            tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
            tgt_idxs[i] = tgt_subtoken_idxs

            tgt_txts[i] = '<q>'.join([' '.join(tt) for tt in tgts[i]])
            # 添加tgt segments idx 其中,[unused99]作为整体的开始符, <q>作为每一个seg的结束符，最后一个seg的结束符为[unused1]
            # 最后一个seg中的内容不需要加，为此需要去掉一个seg内容（因此去掉以结束符结尾的那一段）
            tgt_seg_idxs_[i] = [i for i, t in enumerate(tgt_subtoken_idxs) if t == self.tgt_bos_vid or t == self.tgt_eos_vid or t == self.tgt_sent_split_vid]
            tgt_seg_idxs[i] = [i for i, t in enumerate(tgt_subtoken_idxs) if t == self.tgt_bos_vid or t == self.tgt_sent_split_vid]
            tgt_segs = [tgt_seg_idxs[i][j] - tgt_seg_idxs[i][j-1] for j in range(1, len(tgt_seg_idxs[i])-1)]
            tgt_lens[i] = [1] + tgt_segs
            # 根据长度进行扩充：
            tgt_seg_idxs_[i][0] = -1
            tgt_real_lens_list = [tgt_seg_idxs_[i][j] - tgt_seg_idxs_[i][j - 1] for j in range(1, len(tgt_seg_idxs_[i]))]
            # assert sum(tgt_real_lens_list) == tgt_seg_idxs_[i][-1]
            if i == 0:
                # user_src_tgt_mask : seg_num -> tgt_len
                assert len(tgt_real_lens_list) == len(user_src_tgt_mask)
                user_src_tgt_mask_final = []
                for j in range(len(user_src_tgt_mask)):
                    for k in range(tgt_real_lens_list[j]):
                        user_src_tgt_mask_final.append(user_src_tgt_mask[j])
                assert len(tgt_idxs[i]) == len(user_src_tgt_mask_final)
            else:
                assert len(tgt_real_lens_list) == len(agent_src_tgt_mask)
                agent_src_tgt_mask_final = []
                for j in range(len(agent_src_tgt_mask)):
                    for k in range(tgt_real_lens_list[j]):
                        agent_src_tgt_mask_final.append(agent_src_tgt_mask[j])
                assert len(tgt_idxs[i]) == len(agent_src_tgt_mask_final)

        src_tgt_mask_final = [user_src_tgt_mask_final, agent_src_tgt_mask_final]
        # 记录src每一个段的长度以及tgt每一个seg的长度：
        src_lens = segs
        # topic sequence 部分的操作：
        topic_target = data['topic_target'][:self.args.max_tgt_ntokens]
        aux_seq = data['aux_seq'][:self.args.max_tgt_ntokens]
        
        if is_print:
            # logger.info("src_subtokens: %s", " ".join([str(token) for x in src_subtokens for token in x]))
            logger.info("src_txt: %s", " ".join([str(x) for x in src_txt]))
            logger.info("src_subtoken: %s", " ".join([str(x) for x in src_subtoken_idxs_]))
            logger.info("src_subtoken_idxs: %s", " ".join([str(x) for x in src_subtoken_idxs]))
            logger.info("segments_ids: %s", " ".join([str(x) for x in segments_ids]))
            # cls 也可以作为intent ids
            logger.info("cls_ids: %s", " ".join([str(x) for x in cls_ids]))
            logger.info("tgt_user_subtoken: %s", tgt_subtokens[0])
            logger.info("tgt_agent_subtoken: %s", tgt_subtokens[1])
            logger.info("role_masks: %s", " ".join([str(x) for x in role_masks]))
            logger.info("intent_label: %s", " ".join([str(x) for x in intent_label]))
            logger.info("topic_target: %s", " ".join([str(x) for x in topic_target]))
            logger.info("aux_seq: %s", " ".join([str(x) for x in aux_seq]))
            # print(len(agent_src_tgt_mask_final))
            # print(tgt_real_lens_list)
            # for i in range(2):
            #     for j in range(len(agent_src_tgt_mask_final[i])):
            #         if agent_src_tgt_mask_final[i][j] == 1:
            #             if src_subtoken_idxs[j] == 101:
            #                 print('one')
    
        return src_subtoken_idxs, tgt_idxs, segments_ids, cls_ids, src_txt, tgt_txts, role_masks,\
               intent_label, tgt_seg_idxs, src_lens, tgt_lens,\
               user_utterances, agent_utterances, src_tgt_mask_final, merge_utterances,\
                   sep_src_subtoken_idxs, sep_seg_id, role_label, topic_target, aux_seq, agent_id



def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path)
    for corpus_type in datasets:
        save_path = save_root_path / corpus_type
        save_path.mkdir(parents=True, exist_ok=True)
        a_lst = []
        for fp in (read_root_path / corpus_type).iterdir():
            a_lst.append((corpus_type, fp, args, save_path / f'{fp.stem}.bert.pt'))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, fp, args, save_file = params
    is_test = corpus_type == 'test'
    if (save_file.exists()):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info(f'Processing {fp.stem}' )
    jobs = json.load(fp.open('r', encoding = 'utf-8'))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        # 贪婪算法选取比较合适的句子（ext-abt）
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                        'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpora = {'train': [], 'valid': [], 'test': []}
    
    read_root_path = Path(args.raw_path)
    for corpus_type in ['valid', 'test', 'train']:
        read_path = read_root_path / corpus_type
        for fp in read_path.iterdir():
            corpora[corpus_type].append(fp)
    
    save_root_path = Path(args.save_path)
    for corpus_type in ['train', 'valid', 'test']:
        save_path = save_root_path / corpus_type
        save_path.mkdir(parents=True, exist_ok=True)
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst)):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                with (save_path / f'{p_ct}.json').open('w', encoding='utf-8') as s_f:
                    s_f.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            with (save_path / f'{p_ct}.json').open('w', encoding='utf-8') as s_f:
                # save.write('\n'.join(dataset))
                s_f.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    source, tgt = load_json(f)
    return {'src': source, 'tgt': tgt}


def convert_bio_label(bio_indexes, total_len):
    labels = [0] * total_len
    for (start, end) in bio_indexes:
        labels[start] = 1
        for i in range(start + 1, end + 1):
            labels[i] = 2
    return labels

def sum_split(sum):
    splited_sum = []
    tmp_sum = []
    for w in sum:
        tmp_sum.append(w)
        if w in ['。', '.', '!', '！', ';', '；']:
            splited_sum.append(tmp_sum)
            tmp_sum = []
    if tmp_sum:
        splited_sum.append(tmp_sum)
    return splited_sum


from collections import defaultdict

def DS_format_to_labels(args):
    corpora = {'train': [], 'val': [], 'test': []}

    read_root_path = Path(args.raw_path)

    # label2id = defaultdict()
    label2id = {}
    labels = []
    for corpus_type in ['val', 'test', 'train']:
    # for corpus_type in ['val']:
        read_path = read_root_path / f'{corpus_type}.json'
        with read_path.open('r', encoding='utf-8') as r_f:
            json_data = json.load(r_f)
            if args.data_name == 'MC':
                pass
            else:
                for sample in json_data:
                    for qa_turn in sample['QA']:
                        # if qa_turn['intent'] not in labels:
                        #     labels.append(qa_turn['intent'])
                        # if qa_turn['intent'] not in labels and qa_turn['intent'] != '':
                        #     labels.append(qa_turn['intent'])
                        if 'intent' not in qa_turn.keys():
                            print(qa_turn)
                        if qa_turn['intent'] == '':
                            labels.append('其他')
                        else:
                            labels.append(qa_turn['intent'])
                        # print(qa_turn, qa_turn['intent'])
    # print(set(labels))
    labels = list(filter(None,list(set(labels))))
    print(len(labels))
    # labels = list(set(labels))
    for i, label in enumerate(labels):
        label2id[label] = i

    return label2id


def DS_format_to_lines(args, label2id):

    # 用于确定偏移后的utterance index
    # target_shift = len(label2id) + 2
    target_shift = 3

    corpora = {'train': [], 'val': [], 'test': []}

    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path)
    save_root_path.mkdir(exist_ok=True, parents=True)

    for corpus_type in ['val', 'test', 'train']:
        read_path = read_root_path / f'{corpus_type}.json'
        save_path = save_root_path / f'{corpus_type}.json'

        with read_path.open('r', encoding='utf-8') as r_f:
            json_data = json.load(r_f)
            
            if args.data_name == 'tianchi':
                for pid, sample in json_data.items():
                    # all_summ = []
                    # context = []
                    # role_label = []
                    # 只要非other内容：
                    # 提取summary:
                    # summary1, summary2 = process(sample['report'][0]), process(sample['report'][1])
                    
                    for i in range(2):
                        all_summ = []
                        context = []
                        role_label = []
                        
                        summary = process(sample['report'][i])
                        for sent in sample['dialogue']:
                            tmp_utt = []
                            if sent['dialogue_act'] != 'Other':
                                if sent['speaker'] == '医生':
                                    tmp_utt += ['医生', ':']
                                else:
                                    tmp_utt += ['患者', ':']

                                tmp_utt += list(jieba.cut(sent['sentence']))
                                context.append(tmp_utt)
                                # content.append(sent['speaker'] + ':' + sent['sentence'])
                            
                                if sent['speaker'] == '医生':
                                    role_label.append(1)
                                else:
                                    role_label.append(2)
                    
                        all_summ.extend(summary)

                        corpora[corpus_type].append({'src': context,
                                                    'tgt_all':all_summ,
                                                    'role_label': role_label})
            else:
                # CSDS数据集
                for sample in json_data:
                    flag = 1
                    # for qa_turn in sample['QA']:
                    #     if qa_turn["AnsSummLongUttIDs"] != qa_turn["AnsSummShortUttIDs"]:
                    #         flag=1
                    #         break
                    if flag == 1:
                        user_summ = [list(jieba.cut(sen)) for sen in sample['UserSumm']]
                        agent_summ = [list(jieba.cut(sen)) for sen in sample['AgentSumm']]
                        role_label = []
                        intent_label = []
                        context = []
                        for turn in sample['Dialogue']:
                            tmp_utt = []
                            if args.add_prefix:
                                if turn['speaker'] == 'Q':
                                    tmp_utt += [sample['QRole'], ':']
                                else:
                                    tmp_utt += ['客服', ':']
                            for word in turn['utterance'].split():
                                if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                                    tmp_utt += ['[', word[1:-1], ']']
                                else:
                                    tmp_utt.append(word)
                            context.append(tmp_utt)
                            if turn['speaker'] == 'Q':
                                role_label.append(1)
                            else:
                                role_label.append(2)
                        intent_label = [0] * len(context)
                        user_utterances = []
                        agent_utterances = []
                        merge_utterances = []
                        # 设置用于存放 utterance [start_idi1, end_idi1 ... start_idin, end_idin, Tagi]
                        pairs = []
                        # 对比实验添加全为topic不含有utterance segment id的aux_seq:
                        aux_seq = [] 
                        topic_target = [0]  # 特殊的sos
                        aux_seq = [0]
                        # 进入此topic sequence 循环内
                        for qa_turn in sample['QA']:
                            cur_pair = []
                            # 分别对user和agent进行分segment的utterance ext label识别
                            user_utterance, agent_utterance = [0] * len(context), [0] * len(context)
                            merge_utterance = [0] * len(context)
                            for user_id in qa_turn["QueSummUttIDs"]:
                                user_utterance[user_id] = 1
                                merge_utterance[user_id] = 1
                            for agent_id in qa_turn["AnsSummLongUttIDs"]:
                                agent_utterance[agent_id] = 1
                                merge_utterance[agent_id] = 1
                            user_utterances.append(user_utterance)
                            agent_utterances.append(agent_utterance)
                            merge_utterances.append(merge_utterance)
                            # 先将user和agent的utterance放在一起
                            if qa_turn['intent']:
                                ua_uttids = []
                                for ids in ["QueSummUttIDs","AnsSummLongUttIDs"]:
                                    ua_uttids += [id for id in qa_turn[ids]]

                                for ua_uttid in ua_uttids:
                                    if intent_label[ua_uttid] == 0:
                                        intent_label[ua_uttid] = label2id[qa_turn["intent"]] + 1
                                    else:
                                        # 多分类问题(一个回答或问题可能对应多种意图)
                                        # print(sample['DialogueID'])
                                        # 先考虑随机替换：
                                        if intent_label[ua_uttid] == label2id[qa_turn["intent"]] + 1:
                                            pass
                                        else:
                                            import random 
                                            # 可以设计随机种子记录以后
                                            if random.random() > 0.5: 
                                                intent_label[ua_uttid] = label2id[qa_turn["intent"]] + 1
                            # 针对对应涉及的utterence_id 生成相关的 start与end id 
                            topic_utter_ids = list(set(qa_turn["QueSummUttIDs"]) | set(qa_turn["AnsSummLongUttIDs"]))
                            # print(qa_turn["QueSummUttIDs"])
                            topic_utter_ids.sort()
                            #TODO: Word 格式：
                            for id in topic_utter_ids:
                                cur_pair.extend([id+target_shift])
                            # Span 格式：
                            # 找寻连续或者独立的序列：
                            # start_ids = []
                            # end_ids = []
                            # # 一个队列结构的list用于输出连续性utterance id
                            # queue = []
                            # # print(topic_utter_ids)
                            # for i, idx in enumerate(topic_utter_ids):
                            #     # 初始化第一个元素：
                            #     if i == 0: 
                            #         queue.append(idx)
                            #     else:
                            #         if queue[-1]+1 == idx:
                            #             queue.append(idx)
                            #         else:
                            #             start_ids.append(queue[0])
                            #             end_ids.append(queue[-1])
                            #             queue = []
                            #             queue.append(idx)
                            # start_ids.append(queue[0])
                            # end_ids.append(queue[-1])
                            # assert len(start_ids) == len(end_ids)
                            # # 存在临时list中，加上偏移量
                            # for s,e in zip(start_ids,end_ids):
                            #     cur_pair.extend([s+target_shift,e+target_shift])
                            #     #TODO: 单独将每个utterence id tag分开而不是合在一起作为解码序列：
                            #     # cur_pair.append(label2id['其他' if qa_turn['intent'] == '' or not qa_turn['intent'] else qa_turn['intent']] + 2) 
                            # # 加上tag的标签，加2是由于有shift（sos,eos）
                            # cur_pair.append(label2id['其他' if qa_turn['intent'] == '' or not qa_turn['intent'] else qa_turn['intent']] + 2) 
                            cur_pair.append(2) 
                            pairs.append([p for p in cur_pair])
                            # 只保留topic序列：
                            aux_seq.append(label2id['其他' if qa_turn['intent'] == '' or not qa_turn['intent'] else qa_turn['intent']] + 2)
                        # topic sequence target
                        topic_target.extend(list(chain(*pairs)))
                        topic_target.append(1)  # 特殊的eos
                        aux_seq.append(1)
                        assert len(role_label) == len(intent_label)
                        agent_id = []
                        for idx, qa_turn in enumerate(sample['QA']):
                            if qa_turn["AnsSummLongUttIDs"] != qa_turn["AnsSummShortUttIDs"]:
                                agent_id.append(idx)
                            else:
                                agent_id.append(-1)

                        corpora[corpus_type].append({'src': context,
                                                    'tgt_user': user_summ,
                                                    'tgt_agent': agent_summ,
                                                    'role_label': role_label,
                                                    'intent_label':intent_label,
                                                    'user_utterances':user_utterances,
                                                    'agent_utterances':agent_utterances,
                                                    'merge_utterances':merge_utterances,
                                                    'topic_target':topic_target,
                                                    'aux_seq':aux_seq,
                                                    'agent_id':agent_id})

                        # for idx, qa_turn in enumerate(sample['QA']):
                        #     if qa_turn["AnsSummLongUttIDs"] != qa_turn["AnsSummShortUttIDs"]:
                        #         with open('a_diff','a+',encoding="utf-8") as ft:
                        #             ft.write(str(idx))
                        #             ft.write('\t')
                        #     else:
                        #         ft.write(str(-1))
                        #         ft.write('\t')
                        # ft.write('\n')

        with save_path.open('w', encoding='utf-8') as w_f:
            w_f.write(json.dumps(corpora[corpus_type], indent=4, ensure_ascii=False))


def DS_format_to_bert(pretrained_path, args, tag_mapping):
    corpora = {'train': [], 'val': [], 'test': []}
    bert = BertData(pretrained_path, args, tag_mapping)
    read_root_path = Path(args.raw_path)

    for corpus_type in corpora:
        save_root_path = Path(args.save_path) / corpus_type
        save_root_path.mkdir(exist_ok=True, parents=True)

        is_test = corpus_type[:4] == 'test'
        
        read_path = read_root_path / f'{corpus_type}.json'
        save_path = save_root_path / f'{corpus_type}.bert.bin'

        logger.info(f'Processing {read_path.stem}')
        jobs = json.load(read_path.open('r', encoding='utf-8'))

        for index,d in enumerate(tqdm(jobs)):
            # 调用preprocess处理成bert输入
            # 展示一下具体字典中的数据：
            is_print = False
            if corpus_type[:5] == 'val' and index < 2:
                logger.info("*** Example ***")
                logger.info("guid: %d", index)
                is_print = True
                
            b_data = bert.preprocess(d, is_print, use_bert_basic_tokenizer=True, is_test=is_test)

            if (b_data is None):
                continue
            src_subtoken_idxs, tgt_idxs, segments_ids, cls_ids, src_txt, tgt_txts, role_masks, intent_label, tgt_seg_idxs, src_lens, tgt_lens, user_utterances, agent_utterances, \
                src_tgt_mask_final, merge_utterances, sep_src_subtoken_idxs, sep_seg_id, role_label, topic_target, aux_seq, agent_id = b_data
            # change extractive labels
            b_data_dict = {"src": src_subtoken_idxs,
                           "tgt_user": tgt_idxs[0], "tgt_agent": tgt_idxs[1],
                           "segs": segments_ids, 'clss': cls_ids,
                           "src_txt": src_txt,
                           "tgt_txt_user": tgt_txts[0], "tgt_txt_agent": tgt_txts[1], 
                           'role_mask': role_masks, 'intent_label': intent_label,
                           'tgt_user_seg_idxs': tgt_seg_idxs[0], 'tgt_agent_seg_idxs': tgt_seg_idxs[1],
                           'src_lens':src_lens, 'tgt_user_lens':tgt_lens[0], 'tgt_agent_lens':tgt_lens[1],
                           'user_utterances':user_utterances, 'agent_utterances':agent_utterances,
                           'user_src_tgt_mask_final':src_tgt_mask_final[0],'agent_src_tgt_mask_final':src_tgt_mask_final[1],
                           'merge_utterances':merge_utterances,
                           'sep_src_subtoken_idxs':sep_src_subtoken_idxs, 'sep_seg_id':sep_seg_id,'role_label':role_label,'topic_target':topic_target,'aux_seq':aux_seq,'agent_id':agent_id}

            corpora[corpus_type].append(b_data_dict)

        logger.info('Processed instances %d' % len(corpora[corpus_type]))
        logger.info('Saving to %s' % save_path)
        torch.save(corpora[corpus_type], save_path)
        

def _format_to_bert(params):
    corpus_type, fp, args, save_file = params
    is_test = corpus_type == 'test'
    if (save_file.exists()):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info(f'Processing {fp.stem}' )
    jobs = json.load(fp.open('r', encoding = 'utf-8'))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                        'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
