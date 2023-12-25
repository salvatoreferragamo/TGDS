Code Structure:

```
--- data
   |--- SAMSum
   |--- ...
--- test_save 
--- bart_trainer.py
--- modeling_bart.py
--- README.md
--- requirements.txt
```


Example training on SAMSum with BART:

```
CUDA_VISIBLE_DEVICES=1 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode "baseline" \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--output_dir "test_save"
```
