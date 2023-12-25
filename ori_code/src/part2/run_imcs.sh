export CUDA_VISIBLE_DEVICES=0

# python preprocess_t.py -log_file logs/log -data_name tianchi -max_src_ntokens_per_sent 500 -use_bert_basic_tokenizer False

#bert both for dialoAMC
# python train_t.py -task abs -mode train -bert_data_path data/tianchi/bert/ -dec_dropout 0.2  -model_path output/dialo_part2 -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -lr 0.001 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 0  -log_file logs/bert_none_train.log -finetune_bert True -merge=gate -role_weight=0.5
# python train_t.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/tianchi/bert/ -log_file logs/bert_both__val.log -model_path output/dialo_part2 -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path logs/bert_new_val.txt -temp_dir temp/ -test_all=True -merge=gate

# python train_t.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/tianchi/bert/ -log_file logs/bert_both_ft__test.log -test_from output/dialo_part2/model_step_2400.pt -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 500 -alpha 0.95 -min_length 15 -result_path logs/dialo_part2_own -temp_dir temp/ -merge=gate
