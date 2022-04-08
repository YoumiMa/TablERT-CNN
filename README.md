# TablERT-CNN: Joint Entity and Relation Extraction Based on Table Labeling Using Convolutional Neural Networks

This is the PyTorch code for the paper 'Joint Entity and Relation Extraction Based on Table Labeling Using Convolutional Neural Networks' accepted at [SPNLP2022](http://structuredprediction.github.io/SPNLP22), an ACL 2022 workshop.

This software is implemented based on [SpERT](https://github.com/markus-eberts/spert) [1] and [TablERT](https://github.com/YoumiMa/TablERT) [2].
## Setup

### Requirements

Required
- Python 3.5+ (tested with version 3.7.13)
- PyTorch 1.1.0+ (tested with version 1.11.1)
- transformers 2.2.0+ (tested with version 4.17.0)
- scikit-learn (tested with version 0.21.3)
- tqdm (tested with version 4.64.0)
- numpy (tested with version 1.17.4)

Optional
- jinja2 (tested with version 3.1.1) - if installed, used to export relation extraction examples
- tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard


The details are listed in `requirements.txt`. It is recommended to prepare the enviroment via the following command to reproduce the results.

```
pip install -r requirements.txt

```

## Datasets

### CoNLL04[3]

We have provided the data template in the folder `data/datasets/conll04/`.
We further process the dataset used in [SpERT](https://github.com/markus-eberts/spert)[1]. To obtain the data, first obtain processed SpERT data, then run the pre-processing script as below.

``` 
python3 preprocessing/spert2tabcnn.py $SPERT_DATA_DIR data/datasets/conll04/ 
```


### ACE05

We have provided the data template in the folder `data/datasets/ace05/`.
We further process the dataset used in [DyGIE++](https://github.com/dwadden/dygiepp)[5]. To obtain the data, follow the instructions provided by DyGIE++. Then, run the pre-processing script as below.

``` 
python3 preprocessing/dygiepp2tabcnn.py $SPERT_DATA_DIR data/datasets/conll04/ 
```


### ADE

We have provided the data template in the folder `data/datasets/ade/`.
We further process the dataset used in [SpERT](https://github.com/markus-eberts/spert)[1]. To obtain the data, first obtain processed SpERT data, then run the pre-processing script as below.

``` 
python3 preprocessing/spert2tabcnn.py $SPERT_DATA_DIR data/datasets/ade/

```

Note that the converted dataset has no overlapping entities.

## Experiments

Here we provide examples using the CoNLL04 dataset. Experiments on ACE05 and ADE can be carried out analogously, after putting the pre-processed dataset in corresponding folders.

### Training

To train a model on CoNLL04 train set and evaluate on CoNLL04 development set, run

```
python run.py train --config configs/train_conll04.conf
```

### Evaluation

To evalute a model on CoNLL04 test set, fill in the field `model_path` in `configs/eval_conll04.conf` with the directory of the model. Then, run

```
python run.py eval --config config/eval_conll04.conf
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
[2]Youmi Ma, Tatsuya Hiraoka, and Naoaki Okazaki. Named Entity Recognition and Relation Extraction Using Enhanced Table Filling by Contextualized Representations. 自然言語処理, 29(1):187–223, March 2022. (doi: 10.5715/jnlp.29.187).
[3]Dan Roth and Wen-tau Yih, 2004, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8.
[4]Pankaj Gupta, Hinrich Schütze, and Bernt Andrassy, 2016, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547.
[5]David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi. 2019. Entity, relation, and event extraction with contextualized span representations. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5784–5789.
```


