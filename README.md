# PopCon
This project is a PyTorch implementation of Aggregately Diversified Bundle Recommendation via Popularity Debiasing and Configuration-aware Reranking (PopCon), which is published in PAKDD 2023.

## Overview
The overview of PopCon is as follows.
PopCon consists of two phases, model training phase and reranking phase.
In the training phase, PopCon trains a bundle recommendation model such as DAM or CrossCBR as a backbone while mitigating its popularity bias by a popularity-based negative sampling.
In the raranking phase, PopCon selects candidate bundles for each user and reranks the candidates by a configuration-aware reranking algorithm to maximize both accuracy and aggregate diversity.
For more details, please refer to our paper.
![overview](./overview.png)

## Prerequisties
Our implementation is based on Python 3.8 and Pytorch 1.8.1. Please see the full list of packages required to our codes in `requirements.txt`.

## Datasets
We use 3 datasets in our work: Steam, Youshu, and NetEase.
We include the preprocessed datasets in the repository: `data/{data_name}`.

## Backbone model
We provide DAM, one of the state-of-the-art bundle recommendation models, as a backbone.
It is defined in `models.py`.
CrossCBR, another state-of-the-art model, is available at [https://github.com/mysbupt/CrossCBR](https://github.com/mysbupt/CrossCBR)

## Running the code
You can run the pretraining code by `python pretrain.py` with arguments `--epochs` and `--alpha`.
You can also run the reranking code by `python reranking.py` with arguments `--beta` and `--n`.
To run `reranking.py`, running `pretrain.py` must precede because it returns a recommendation results of a model.
We provide `demo.sh`, which reproduces the experiments of our work.

## Citation
Please cite this paper when you use our code.
```
@inproceedings{conf/pakdd/JeonKLLK23,
  author    = {Hyunsik Jeon and
               Jongjin Kim and
               Jaeri Lee and
               Jong-eun Lee and
               U Kang},
  title     = {Aggregately Diversified Bundle Recommendation via Popularity Debiasing and Configuration-aware Reranking},
  booktitle = {PAKDD},
  year      = {2023},
}
```

## License
This software may be used only for non-commercial purposes (e.g., research evaluation) in universities.
Please contact Prof. U Kang (ukang@snu.ac.kr) if you want to use it for other purposes or use it in places other than universities.
