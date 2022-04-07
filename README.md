# TablERT-CNN: Joint Entity and Relation Extraction Based on Table Labeling Using Convolutional Neural Networks

This is the PyTorch code for the paper 'Joint Entity and Relation Extraction Based on Table Labeling \\Using Convolutional Neural Networks' accepted at [SPNLP2022](http://structuredprediction.github.io/SPNLP22), an ACL 2022 workshop.

This software is implemented based on [SpERT](https://github.com/markus-eberts/spert) [1] and [TablERT](https://github.com/YoumiMa/TablERT) [2].
## Setup

Requirments are listed in `requirements.txt`. It is recommended to prepare the enviroment via the following command to reproduce the results.

```
pip install -r requirements.txt

```

## Datasets

### CoNLL04

We release the preprocessed CoNLL04 [3] data in the folder `data/datasets/conll04/`. The data split follow that of [1], [2] and [4].

### ACE05



### ADE

obtained by pre-processing those of SpERT[1].

## Experiments

Here we provide examples using the CoNLL04 dataset. Experiments on ACE05 and ADE can be carried out analogously, after putting the pre-processed dataset in corresponding folders.

### Training

To train a model on CoNLL04 train set and evaluate on CoNLL04 development set, run

```
python ./run.py train --config configs/train_conll04.conf
```

### Evaluation

To evalute a model on CoNLL04 test set, fill in the field `model_path` in `configs/eval_conll04.conf` with the directory of the model. Then, run

```
python ./run.py eval --config config/eval_conll04.conf
```

## Citation

If you use the provided code in your work, please cite the following paper:

```
@inproceedings{Ma:SPNLP2022,
    title = "Joint Entity and Relation Extraction Based on Table Labeling Using Convolutional Neural Networks",
    author={Youmi Ma and Tatsuya Hiraoka and Naoaki Okazaki},
    booktitle = "6th Workshop on Structured Prediction for NLP (SPNLP)",
    month = may,
    year = "2022",
    pages = "(to appear)",
    address = "Dublin",
    publisher = "Association for Computational Linguistics",
}
```

## References
```
[1]Markus Eberts and Adrian Ulges, 2020, 'Span-based joint entity and relation extraction with transformerpre-training' In 24th European Conference on Artifi-cial Intelligence (ECAI).
[2]
[3]Dan Roth and Wen-tau Yih, 2004, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8.
[4]Pankaj Gupta, Hinrich Schütze, and Bernt Andrassy, 2016, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547.
[5]Heike Adel and Hinrich Schütze, 2017, 'Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification', EMNLP 2017, pp. 1723--1729. 
```


