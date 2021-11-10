import argparse
import math
import os

import torch
from torch import nn
from torch.optim import Optimizer


import transformers
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel

from src import models
from src.entities import Dataset
from src.evaluator import Evaluator
from src.input_reader import JsonInputReader, BaseInputReader
from src.loss import TableLoss, Loss

# from src.parallel import DataParallelModel, DataParallelCriterion

from tqdm import tqdm
from src.trainer import BaseTrainer
from src import util
from src import sampling

from typing import List

from pytorch_memlab import MemReporter

import math

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def align_label(entity: torch.tensor, rel: torch.tensor, token_mask: torch.tensor):
    """ Align tokenized label to word-piece label, masked by token_mask. """

    batch_size = entity.shape[0]
    token_count = token_mask.to(torch.bool).sum()
    batch_entity_labels = []
    batch_rel_labels = []

    for b in range(batch_size):
        batch_entity_labels.append(torch.masked_select(entity[b], token_mask[b].sum(dim=0).to(torch.bool)))
        rel_ = rel[b][token_mask[b].sum(dim=0).to(torch.bool)]
        batch_rel_labels.append(rel_.t()[token_mask[b].sum(dim=0).to(torch.bool)].t())
    return batch_entity_labels, batch_rel_labels



class TableFTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                             cache_dir=args.cache_path)
        

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')


        self._best_results['ner_micro_f1'] = 0
        self._best_results['rel_micro_f1'] = 0
        self._best_results['rel_ner_micro_f1'] = 0

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        train_dataset = input_reader.read(train_path, train_label)
        valid_dataset = input_reader.read(valid_path, valid_label) 
        self._log_datasets(input_reader)
        
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # load model
        model = self._load_model(input_reader)
        model.to(self._device)
#         print("devices:", self._gpu_count)
#         model = nn.DataParallel(model) 
       
        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        
        
        # create scheduler

        if args.scheduler == 'constant':
            scheduler = transformers.get_constant_schedule(optimizer)
        elif args.scheduler == 'constant_warmup':
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total)
        elif args.scheduler == 'linear_warmup':
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total)
        elif args.scheduler == 'cosine_warmup':
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total)            
        elif args.scheduler == 'cosine_warmup_restart':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total,
                                                                     num_cycles= args.num_cycles)            


        # create loss function
        rel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        compute_loss = TableLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, compute_loss, validation_dataset, input_reader, 0, updates_epoch)

        reporter = MemReporter(model)
#         print("============ before bw ===============")
#         reporter.report()        
        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch, input_reader.entity_label_count, input_reader.relation_label_count)
            

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                ner_acc, rel_acc, rel_ner_acc = self._eval(model, compute_loss, validation_dataset, input_reader, epoch, updates_epoch)     
                if args.save_best:
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                    self._save_best(model=model, optimizer=optimizer if self.args.save_optimizer else None, 
                        accuracy=ner_acc[2], iteration=epoch * updates_epoch, label='ner_micro_f1', extra=extra)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)


    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label) 
        self._log_datasets(input_reader)

        # create model
        model = self._load_model(input_reader)
        model.to(self._device)
#        print("devices:", self._gpu_count)
#         model = nn.DataParallel(model) 

        # create loss function
        rel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if args.model_type == '2d_trans':
            compute_loss = TableLoss(rel_criterion, entity_criterion, model)

        # evaluate
        self._eval(model, compute_loss, input_reader.get_dataset(dataset_label), input_reader)
        self._logger.info("Logged in: %s" % self._log_path)

        
        
        
    def _load_model(self, input_reader):
        model_class = models.get_model(self.args.model_type)

        config = AutoConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path, output_hidden_states=True)
        
        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            config = config,
                                            # src model parameters
                                            entity_labels = input_reader.entity_label_count,
                                            relation_labels = input_reader.relation_label_count,
                                            entity_label_embedding=self.args.entity_label_embedding,
                                            rel_label_embedding = self.args.rel_label_embedding,
                                            pos_embedding = self.args.pos_embedding,
                                            encoder_embedding = self.args.encoder_embedding,
                                            encoder_hidden = self.args.encoder_hidden,
                                            encoder_heads = self.args.encoder_heads,
                                            encoder_layers = self.args.encoder_layers,
                                            attn_type = self.args.attn_type,
                                            prop_drop = self.args.prop_drop,
                                            freeze_transformer=self.args.freeze_transformer,
                                            device=self._device)
        return model


    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, 
                    optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, 
                     entity_labels_count:int, relation_labels_count: int):
        
        self._logger.info("Train epoch: %s" % epoch)
        
        dataset.switch_mode(Dataset.TRAIN_MODE)

        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        global_iteration = epoch * updates_epoch
        total = dataset.document_count // self.args.train_batch_size


        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):


            model.train()
            batch = util.to_device(batch, self._device)
#             print([self._tokenizer.convert_ids_to_tokens(encoding) for encoding in batch['encodings']])
#             entity_labels, rel_labels = align_label(batch['ent_labels'], batch['rel_labels'], batch['start_token_masks'])
#             pred_entity_labels, pred_rel_labels = align_label(batch['pred_ent_labels'], batch['pred_rel_labels'], batch['start_token_masks'])
            ent_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['ctx_masks'],
                                           token_masks=batch['token_masks'],
                                           token_context_masks=batch['token_ctx_masks'], 
                                           entity_masks=batch['pred_ent_masks'],
                                           bert_layer = self.args.bert_layer,
                                           pred_entities=batch['pred_ent_labels'],
                                           pred_relations=batch['pred_rel_labels'])

            loss = compute_loss.compute(ent_logits, batch['ent_labels'], rel_logits, batch['rel_labels'], batch['token_ctx_masks']) 

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, compute_loss: Loss, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

#         if isinstance(model, DataParallel):
#             # currently no multi GPU support during evaluation
#             model = model.module

        # create evaluator
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}.json')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                            self.args.model_type, self.args.example_count,
                            self._examples_path, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                 drop_last=False, num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)
        
        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)
                torch.save([self._tokenizer.decode(encoding, clean_up_tokenization_spaces=False).split() for encoding in batch['encodings']], 'words')
#                 print(batch['ent_labels'])
#                 print(batch['rel_labels'])
                # run model (forward pass)
#                 entity_labels, rel_labels = align_label(batch['ent_labels'], batch['rel_labels'], batch['start_token_masks'])
#                 pred_entity_labels, pred_rel_labels = align_label(batch['pred_ent_labels'], batch['pred_rel_labels'], batch['start_token_masks'])

                ent_logits, rel_logits = model(encodings=batch['encodings'],
                                               context_masks=batch['ctx_masks'],
                                               token_masks=batch['token_masks'],
                                               token_context_masks=batch['token_ctx_masks'],
                                               entity_masks=batch['pred_ent_masks'],
                                               bert_layer = self.args.bert_layer,
                                               pred_entities=batch['pred_ent_labels'],
                                               pred_relations=batch['pred_rel_labels'])

                loss = compute_loss.compute(ent_logits, batch['ent_labels'], rel_logits, batch['rel_labels'], batch['token_ctx_masks'], is_eval=True) 

                evaluator.eval_batch(ent_logits, rel_logits, batch)


        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_ner_eval = evaluator.compute_scores()
        
        self._log_eval(*ner_eval, *rel_eval, *rel_ner_eval, loss,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_examples:
            evaluator.store_examples()

        return ner_eval, rel_eval, rel_ner_eval

    def _get_optimizer_params(self, model):
        
        params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and not 'bert' in n],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay) and not 'bert' in n], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr_bert},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay': 0.0, 'lr': self.args.lr_bert}]
        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss[0], global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss[0], global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss.tolist(), epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss.tolist(), epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_ner_prec_micro: float, rel_ner_rec_micro: float, rel_ner_f1_micro: float,
                  rel_ner_prec_macro: float, rel_ner_rec_macro: float, rel_ner_f1_macro: float,
                  loss: List[torch.Tensor], epoch: int, iteration: int, global_iteration: int, label: str):


        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_ner_prec_micro', rel_ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_micro', rel_ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_micro', rel_ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_prec_macro', rel_ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_macro', rel_ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_macro', rel_ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'loss', loss[0], global_iteration)


        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_ner_prec_micro, rel_ner_rec_micro, rel_ner_f1_micro,
                      rel_ner_prec_macro, rel_ner_rec_macro, rel_ner_f1_macro,
                      loss.tolist(), epoch, iteration, global_iteration)



    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)
        self._logger.info("Context size: %s" % input_reader.context_size)
        
        self._logger.info("Entities:")
        for e in input_reader.entity_labels.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_labels.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))
        
        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)
            self._logger.info("Token count: %s"% d.token_count)
            self._logger.info("Maximum sequence length: %s"% d.max_seq_len)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_ner_prec_micro', 'rel_ner_rec_micro', 'rel_ner_f1_micro',
                                                 'rel_ner_prec_macro', 'rel_ner_rec_macro', 'rel_ner_f1_macro',
                                                 'loss', 'epoch', 'iteration', 'global_iteration']})
