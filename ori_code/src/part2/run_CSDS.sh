export CUDA_VISIBLE_DEVICES=0


# python preprocess.py -log_file logs/log -data_name CSDS -use_bert_basic_tokenizer False

# python train.py -task abs -mode train -bert_data_path data/CSDS/bert/ -dec_dropout 0.2  -model_path output/csds_part2 -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -lr 0.001 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 0  -log_file logs/csds.log -finetune_bert True -merge=gate -role_weight=0.5
# python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/CSDS/bert/ -log_file logs/topic_val.log -model_path output/csds_part2 -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path logs/topic_val.txt -temp_dir temp/ -test_all=True -merge=gate
# python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/CSDS/bert/ -log_file logs/bert_both_ft__test.log -test_from output/csds_part2/model_step_1600.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 400 -alpha 0.95 -min_length 15 -result_path logs/csds_part2 -temp_dir temp/ -merge=gate
