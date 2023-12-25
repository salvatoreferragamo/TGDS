# TGDS

Pytorch implementation of our paper: [Let Topic Flow: A Unified **T**opic-**G**uided Segment-wise **D**ialogue **S**ummarization Framework].

The code is partially referred to [BERTAbs](https://github.com/nlpyang/PreSumm) and [BERT-both](https://github.com/xiaolinAndy/RODS).

## Requirements

* Python 3.6 or higher
* torch==1.10.0+cu113
* transformers==4.12.3
* torchtext==0.5.0
* rouge==1.0.1
* tensorboardX==2.5.1
* nltk==3.7
* jieba==0.42.1
* numpy==1.19.5

## Environment

* Nvidia Ge-Force RTX-3090 Graphical Card with 24G graphical memory
* CUDA 11.6

## Usage

Note that: For the property rights protection and considering the ease of operation, we only release our **TGDS-pipeline** code on the framework of **BERTAbs** for examination, which can beat the state-of-the-art on CSDS and IMCS-21.

All the code will be released in github after accepted, plz waiting...

1. Download BERT checkpoints and datasets.

   The pretrained BERT checkpoints can be found at:

   * Chinese BERT: [https://github.com/ymcui/Chinese-BERT-wwm]()

   Both datasets can be found in data folder.

   Put BERT checkpoints into the directory **pretrained** and put the **data** folder into **part1** and **part2** like this:

   ```
   --- part1
   |--- data
      |--- CSDS
      |--- IMCS-21
   |--- ...
   --- part2
   |--- data
      |--- CSDS
      |--- IMCS-21
   |--- ...
   --- pretrained
   |--- bert_base_chinese
      |--- config.json
      |--- pytorch_model.bin
      |--- vocab.txt
   ```
2. Part1 (**Topic Guidance Generator**)

   Run the bash file *run_CSDS.sh* or *run_dialo.sh* in part1 folder to train and test.
3. Part2 (**Segmentation-based Summarizer**)

   Put the Part1's results (stores in **\logs** in Part1 folder) to **\logs** in Part2 folder as the topic guidance.

   Run the bash file *run_CSDS.sh* or *run_dialo.sh* in part2 folder to train and test.
4. Evaluation

   Test Mode in the bash file *run_CSDS.sh* or *run_imcs.sh* ouput the ROUGE-score which can demonstrate the effectiveness of our method.

   We also provide the prediction results of our method for fast verification.

   ```
   |--- csds.gold
   |--- TGDS.csds.candidate
   |--- TGDS(with part1 gold label).csds.candidate

   |--- dialo.gold
   |--- TGDS.dialo.candidate
   |--- TGDS(with part1 gold label).dialo.candidate
   ```
