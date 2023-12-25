export CUDA_VISIBLE_DEVICES=1

# python preprocess_t.py -log_file logs/log -data_name tianchi -max_src_ntokens_per_sent 500 -use_bert_basic_tokenizer False

# python train_t.py -task abs -mode train -bert_data_path data/tianchi/bert/ -dec_dropout 0.2  -model_path output/dialo_part1 -sep_optim true -lr_bert 0.001 -lr_dec 0.01 -save_checkpoint_steps 500 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 1  -log_file logs/topic.log -finetune_bert True -merge=gate -role_weight=0.5 -kl_weight=0 -share_emb
# python train_t.py -task abs -mode validate -batch_size 1 -test_batch_size 1 -bert_data_path data/tianchi/bert/ -log_file logs/topic_val.log -model_path output/dialo_part1 -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path logs/topic_val.txt -temp_dir temp/ -test_all=True -merge=gate
# python train_t.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/tianchi/bert/ -log_file logs/bert_both_ft__test.log -test_from output/dialo_part1/model_step_3000.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 15 -result_path logs/bert_dialo -temp_dir temp/ -merge=gate
