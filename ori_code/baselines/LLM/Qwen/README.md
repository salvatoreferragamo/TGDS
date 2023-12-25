# Qwen-7B

Considering both CSDS and IMCS-21 are Chinese datasets, we choose to perform instruction tuning with **Qwen-7B** model as the initial checkpoint.

We use [Firefly](https://github.com/yangjianxin1/Firefly) as the framework of finetuning and choose **QLORA** for parameter efficient training.

```
pip install requirements.txt
```

Input:

We preprocess the datasets (in the fokder of "./data/csds or ./data/imcs") into the input format of Firefly:

```
<s>input1</s>target1</s>input2</s>target2</s>...
```

Fintuning with Qlora:

```
torchrun --nproc_per_node={num_gpus} train_qlora.py --train_args_file train_args/qlora/qwen-7b-sft-qlora.json
```

Evaluate:

```
cd script/evaluate
python generate_answer.py
```
