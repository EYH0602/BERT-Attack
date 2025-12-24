# BERT-ATTACK

Code for our EMNLP2020 long paper:

*[BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://arxiv.org/abs/2004.09984)*



## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.4.0
- [transformers](https://github.com/huggingface/transformers) 2.9.0
- [TextFooler](https://github.com/jind11/TextFooler)




## Usage

To train a classification model, please use the run_glue.py script in the huggingface transformers==2.9.0.

To generate adversarial samples based on the masked-LM, run

```
python bertattack.py --data_path data_defense/imdb_1k.tsv --mlm_path bert-base-uncased --tgt_path models/imdbclassifier --use_sim_mat 1 --output_dir data_defense/imdb_logs.tsv --num_label 2 --use_bpe 1 --k 48 --start 0 --end 1000 --threshold_pred_score 0
```

* --data_path: We take IMDB dataset as an example. Datasets can be obtained in [TextFooler](https://github.com/jind11/TextFooler).
* --mlm_path: We use BERT-base-uncased model as our target masked-LM.
* --tgt_path: We follow the official fine-tuning process in [transformers](https://github.com/huggingface/transformers) to fine-tune BERT as the target model.
* --k 48: The threshold k is the number of possible candidates 
* --output_dir : The output file.
* --start:  --end: in case the dataset is large, we provide a script for multi-thread process.
* --threshold_pred_score: a score in cutting off predictions that may not be suitable (details in Section5.1)


## Note

The datasets are re-formatted to the GLUE style. 

Some configs are fixed, you can manually change them.

If you need to use similar-words-filter, you need to download and process consine similarity matrix following [TextFooler](https://github.com/jind11/TextFooler). We only use the filter in sentiment classification tasks like IMDB and YELP.

If you need to evaluate the USE-results, you need to create the corresponding tensorflow environment [USE](https://tfhub.dev/google/universal-sentence-encoder/4).

For faster generation, you could turn off the BPE substitution.

As illustrated in the paper, we set thresholds to balance between the attack success rate and USE similarity score.

The multi-thread process use the batchrun.py script

You can run 

```
cat cmd.txt | python batchrun.py --gpus 0,1,2,3 
```

to simutaneously generate adversarial samples of the given dataset for faster generation.
We use the IMDB dataset as an example. 

## ContentFuzz integration

We added a helper script to run BERT-Attack directly on the ContentFuzz stance datasets (C-STANCE-A/B, SemEval16, VAST) and summarize results.

Example commands (from repo root):

```
# C-STANCE-A (Chinese BERT classifier + MLM)
uv run src/run_bert_attack.py c-stance-a --tgt-model saved_models/hfl/chinese-bert-wwm --mlm-model hfl/chinese-bert-wwm

# SemEval16 (English BERT classifier + MLM)
uv run src/run_bert_attack.py sem16 --tgt-model saved_models/google-bert/bert-base-uncased/sem16 --mlm-model google-bert/bert-base-uncased
```

Key flags:
- `--tgt-model`: fine-tuned classifier checkpoint to attack.
- `--mlm-model`: MLM used for substitutions (defaults to the dataset’s default if omitted).
- `--sample-n`: randomly attack a subset.
- `--start/--end`: slice the dataset.
- `--use-sim-mat --embed-path --sim-mat-path`: enable cosine-sim filtering with counter-fitted vectors.

Outputs:
- Raw attack logs: `results/bert_attack/bert-attack+{model}+{dataset}.json`
- Metrics summary: `results/bert_attack/bert-attack+{model}+{dataset}.summary.json`
