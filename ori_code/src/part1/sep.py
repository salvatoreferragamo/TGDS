from cal_rouge import cal_rouge_path
cal_rouge_path('/home/sda/hanqinyu/DPC/RODS-main/models/BERT_interact_former/logs/bert_gen_val.3000.candidate', '/home/sda/hanqinyu/DPC/RODS-main/models/BERT_interact_former/logs/bert_gen_val.3000.gold')


# def get_sents_str(file_path):
#     sents = []
#     with open(file_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip().replace(' ','').replace('<q>','').replace('0','')
#             sents.append(line)
#     return sents

# # #run bleu
# from nltk.translate.bleu_score import corpus_bleu

# pred_file, ref_file = 'can——.txt', 'gold——.txt'
# refs = get_sents_str(ref_file)
# preds = get_sents_str(pred_file)

# bleu_preds = [list(s) for s in preds]
# bleu_refs = [[list(s)] for s in refs]
# bleu_score = corpus_bleu(bleu_refs, bleu_preds)
# print('BLEU: ', bleu_score)