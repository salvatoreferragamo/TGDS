#encoding=utf-8


import argparse
import time
import json

from others.logging import init_logger
from prepro import data_builder
# from prepro import data_builder_t


def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())

def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())



def do_format_xsum_to_lines(args):
    print(time.clock())
    data_builder.format_xsum_to_lines(args)
    print(time.clock())

def do_tokenize(args):
    print(time.clock())
    data_builder.tokenize(args)
    print(time.clock())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-select_mode", default='greedy', type=str)
    # parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../data/CNN-DM/line_data')
    parser.add_argument("-save_path", default='../../data/CNN-DM/bert_data')

    parser.add_argument("-seed", default=2526, type=int)
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=0, type=int)
    parser.add_argument('-max_src_nsents', default=80, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=0, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=0, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    # 划分为intent段落
    parser.add_argument('-max_tgt_seg_ntokens', default=50, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-is_dialogue", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-add_prefix", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default='../logs/cnndm.log')

    parser.add_argument('-dataset', default='')

    parser.add_argument('-n_cpus', default=4, type=int)
    parser.add_argument('-data_name', default='CSDS', type=str)


    args = parser.parse_args()
    init_logger(args.log_file)


    # 处理文本
    # label2id = {"主诉":0,"现病史":1,"辅助检查":2,"既往史":3,"诊断":4,"建议":5}
    # args.raw_path = 'data/' + args.data_name + '/'
    # args.save_path = 'data/' + args.data_name + '/line/'
    # data_builder_t.DS_format_to_lines(args, label2id)

    # 处理成bert输入的格式
    # args.raw_path = 'data/' + args.data_name + '/line/'
    # args.save_path = 'data/' + args.data_name + '/bert/'
    # pretrained_path = '../pretrained/bert_base_chinese'

    # 增加Tag token:
    # label2tag = {}
    # for key, value in label2id.items():
    #     # print(key, value)
    #     if key == '':
    #         key = '其他'
    #     label2tag[key] = '<<' + key + '>>'
    # data_builder_t.DS_format_to_bert(pretrained_path, args, label2tag)
    # data_builder_t.DS_format_to_bert(pretrained_path, args)

    # label2id_path = 'data/' + args.data_name + '/dict/label2id.json'
    # label2tag_path = 'data/' + args.data_name + '/dict/label2tag.json'
    # label2vid_path = 'data/' + args.data_name + '/dict/label2vid.json'

    # import json
    # label2id_str = json.dumps(label2id,indent=4,ensure_ascii=False)
    # with open(label2id_path, 'w') as json_file:
    #     json_file.write(label2id_str)
    # label2tag_str = json.dumps(label2tag,indent=4,ensure_ascii=False)
    # with open(label2tag_path, 'w') as json_file:
    #     json_file.write(label2tag_str)
    
    # from others.tokenization import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True, mapping=label2tag)
    # # 取出每个tag对应的vocab_id:
    # label2vid = {}
    # for key, value in label2tag.items():
    #     key_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value))
    #     # print(key_id)
    #     assert len(key_id) == 1, value
    #     label2vid[value] = key_id[0] 
    # label2vid_str = json.dumps(label2vid,indent=4,ensure_ascii=False)
    # with open(label2vid_path, 'w') as json_file:
    #     json_file.write(label2vid_str)

    
    args.raw_path = 'data/' + args.data_name + '/'
    label2id = data_builder.DS_format_to_labels(args)
    # print(label2id)
    # 处理文本
    args.raw_path = 'data/' + args.data_name + '/'
    args.save_path = 'data/' + args.data_name + '/line/'
    data_builder.DS_format_to_lines(args, label2id)

    # 处理成bert输入的格式
    args.raw_path = 'data/' + args.data_name + '/line/'
    args.save_path = 'data/' + args.data_name + '/bert/'
    pretrained_path = '../pretrained/bert_base_chinese'
    # 增加Tag token:
    label2tag = {}
    for key, value in label2id.items():
        # print(key, value)
        if key == '':
            key = '其他'
        label2tag[key] = '<<' + key + '>>'
    data_builder.DS_format_to_bert(pretrained_path, args, label2tag)
    # label2id 以及 label2tag也要写入文件中以备用：
    label2id_path = 'data/' + args.data_name + '/dict/label2id.json'
    label2tag_path = 'data/' + args.data_name + '/dict/label2tag.json'
    label2vid_path = 'data/' + args.data_name + '/dict/label2vid.json'
    import json
    label2id_str = json.dumps(label2id,indent=4,ensure_ascii=False)
    with open(label2id_path, 'w') as json_file:
        json_file.write(label2id_str)
    label2tag_str = json.dumps(label2tag,indent=4,ensure_ascii=False)
    with open(label2tag_path, 'w') as json_file:
        json_file.write(label2tag_str)

    from others.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True, mapping=label2tag)
    # 取出每个tag对应的vocab_id:
    label2vid = {}
    for key, value in label2tag.items():
        key_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value))
        assert len(key_id) == 1, value
        label2vid[value] = key_id[0] 
    label2vid_str = json.dumps(label2vid,indent=4,ensure_ascii=False)
    with open(label2vid_path, 'w') as json_file:
        json_file.write(label2vid_str)
        

    # 统计训练集中intent分布：
    # with open('data/CSDS/dict/label2id.json','r',encoding = 'utf-8') as js_object:
    #     label2id = json.load(js_object) 
    # label2total ={}
    # for key,value in label2id.items():
    #     label2total[key] = 0
    # read_path = 'data/CSDS/train.json'
    # turn_st = {}
    # for i in range(8):
    #     turn_st[i+1] = 0
    # with open(read_path, 'r', encoding='utf-8') as r_f:
    #         json_data = json.load(r_f)
    #         max_turn = 0
    #         for sample in json_data:
    #             cur = 0
    #             for qa_turn in sample['QA']:
    #                 cur += 1
    #                 # if qa_turn['intent'] not in labels:
    #                 #     labels.append(qa_turn['intent'])
    #                 # if qa_turn['intent'] not in labels and qa_turn['intent'] != '':
    #                 #     labels.append(qa_turn['intent'])
    #                 if qa_turn['intent'] == '' or not qa_turn['intent']:
    #                     label2total['其他'] += 1
    #                 else:
    #                     label2total[qa_turn['intent']] += 1
    #             max_turn = max(max_turn,cur)
    #             turn_st[cur] += 1
    # print(label2total)
    # print(max_turn)
    # print(turn_st)
