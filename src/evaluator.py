import os
import warnings
from typing import List, Tuple, Dict

import torch
import torch.nn as nn

import math, codecs, json
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import AutoTokenizer

from src.entities import Document, Dataset, EntityLabel, EntityType
from src.input_reader import JsonInputReader
from src.opt import jinja2
from src.util import subset_of


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))



class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: AutoTokenizer,
                 model_type: str, example_count: int, example_path: str, 
                 epoch: int, dataset_label: str, max_epoch: int = 0):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._model_type = model_type

        self._max_epoch = max_epoch
        self._epoch = epoch
        self._dataset_label = dataset_label

        self._example_count = example_count
        self._examples_path = example_path

        # relations
        self._gt_relations = []  # ground truth
        self._prev_pred_relations = []
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._prev_pred_entities = []
        self._pred_entities = []  # prediction

        self._pseudo_entity_label = EntityLabel('Entity', 1, 'Entity', 'Entity')
        self._pseudo_entity_type = EntityType([self._pseudo_entity_label], 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)
        


    def eval_batch(self, batch_ent_logits: torch.tensor, 
                   batch_rel_logits: torch.tensor, batch: dict):
        
        batch_size = batch_ent_logits.shape[0]
        # get model predictions for sample
       
        for i in range(batch_size):
            ent_logits = batch_ent_logits[i]
            rel_logits = batch_rel_logits[i]

            # select highest score at each cell
            ent_scores, ent_preds = torch.softmax(ent_logits,dim=-1).max(dim=-1)
            rel_scores, rel_preds = torch.softmax(rel_logits,dim=-1).max(dim=-1)
            # print("gold labels:{}\n pred labels:{}".format(batch['ent_labels'], ent_preds))
            pred_entities = self._convert_pred_entities_end(ent_preds, ent_scores, batch['token_ctx_masks'][i], batch['token_masks'][i])  # token_ctx_masks: length in token        
            # print("pred entities:", pred_entities)
            # pred_relations = self._convert_pred_relations_(rel_logits, pred_entities, batch['token_ctx_masks'][i], batch['token_masks'][i])
            pred_relations = self._convert_pred_relations_upper(rel_preds, rel_scores, pred_entities, batch['token_masks'][i]) 
            self._pred_entities.append(pred_entities)
            self._pred_relations.append(pred_relations)    
#             print("pred:", pred_entities)
#             print("gold:", self._gt_entities[i])
    
    def compute_scores(self):

        print("Evaluation")
        print("")
        print("--- Entities (NER) ---")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without NER")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With NER")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=True)
        rel_ner_eval = self._score(gt, pred, print_results=True)
        return ner_eval, rel_eval,rel_ner_eval

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        rel_examples = []
        rel_examples_ner = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            json_entities = self._dump_json(doc, self._gt_entities[i], self._pred_entities[i],
                                              self._gt_relations[i], self._pred_relations[i])
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # relations
            # without entity types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_entity_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with entity types
            rel_example_ner = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_entity_types=True, to_html=self._rel_to_html)
            rel_examples_ner.append(rel_example_ner)

        label, epoch = self._dataset_label, self._epoch

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')

        # relations
        # without entity types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_sorted', label, epoch),
                             template='relation_examples.html')

        # with entity types
        self._store_examples(rel_examples_ner[:self._example_count],
                             file_path=self._examples_path % ('rel_ner', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_ner[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_ner_sorted', label, epoch),
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]
            sample_gt_entities = [entity.as_tuple_span() for entity in gt_entities]
            self._gt_relations.append(sample_gt_relations)
            self._gt_entities.append(sample_gt_entities)
    
        
        
    def _convert_pred_entities_end(self, pred_types: torch.tensor, pred_scores: torch.tensor, 
                                token_ctx_mask: torch.tensor, token_mask: torch.tensor):
        #### for word-level.
        converted_preds = []
        
        encoding_length = token_ctx_mask.long().sum() + 1
        curr_type = 0
        start = 1
        
        for i in range(encoding_length):
            
            curr_token = token_mask[i].nonzero()
            type_idx = pred_types[i].item()

            if token_ctx_mask[i]:
                
                if type_idx == 0:
                    start = curr_token[-1].item() + 1
                    continue
                # first element of subtoken span
                score = pred_scores[i].item()
                curr_type = math.ceil(type_idx/4)
                is_end = type_idx % 2 == 0

                if is_end:
                    end = curr_token[-1].item() + 1
                    entity_type = self._input_reader.get_entity_type(curr_type)
                    converted_pred = (start, end, entity_type, score)
                    converted_preds.append(converted_pred)                
                    start = end
               
                  

        return converted_preds


    def _convert_pred_entities_start(self, pred_types: torch.tensor, pred_scores: torch.tensor, 
                                token_mask: torch.tensor):

        converted_preds = []
        encoding_length = token_mask.shape[0]
        curr_type = 0
        start = 1

        for i in range(pred_types.shape[0]):
            curr_token = token_mask[i+1][1:encoding_length-1].nonzero()
            type_idx = pred_types[i].item()
            score = pred_scores.item()
            
            is_start = type_idx % 4 == 1 or type_idx % 4 == 2 or type_idx == 0
            
            if is_start and curr_type != 0:
                # every time encounters a start entity, update the entity list once
                end = curr_token[0].item() + 1
                converted_pred = (start, end, entity_type, score)
                converted_preds.append(converted_pred)                
                start = curr_token[0].item() + 1              
            
            # update the BILOU label of current token
            curr_type = math.ceil(type_idx/4)
            entity_type = self._input_reader.get_entity_type(curr_type)  


            if type_idx == 0:
                start = curr_token[-1].item() + 2
        
        if curr_type != 0: # last word in the sentence is included in an entity span
                converted_pred = (start, curr_token[-1].item() + 2, entity_type, score)
                converted_preds.append(converted_pred)             


        return converted_preds

    def _convert_pred_relations_(self, pred_logits: torch.tensor, pred_entities: List[tuple], 
                                 token_ctx_mask: torch.tensor, token_mask: torch.tensor):
        
        converted_rels = []
        num_labels = pred_logits.shape[-1] 
        encoding_length = token_ctx_mask.long().sum() + 1

        forward_labels = [0] + [i for i in range(1, num_labels, 2)]
        backward_labels = [0] + [i for i in range(2, num_labels, 2)]
        forward_logits = pred_logits[:, :, forward_labels]
        backward_logits = pred_logits[:, :, backward_labels]
        scores = forward_logits + backward_logits.transpose(0,1)
#         scores = pred_logits
#         print("forward:", forward_logits, forward_logits.argmax(dim=0))
#         print("backward:", backward_logits, backward_logits.argmax(dim=0))
#         print("scores as head:", scores.argmax(dim=0))
        
        for head in pred_entities:
#             head_span = range(head[0], head[1])
            head_span = self._get_token_span(head, token_mask, encoding_length)
            for tail in pred_entities:
#                 tail_span = range(tail[0], tail[1])
                tail_span = self._get_token_span(tail, token_mask, encoding_length)

                score = scores[head_span[0]:head_span[-1]+1, tail_span[0]:tail_span[-1]+1, :]
                score = torch.softmax(score.sum(dim=0).sum(dim=0), dim=-1)
                pred_score, rel = score.max(dim=-1)

                if rel.item() and head != tail: # forward
                    pred_head_type = head[2]
                    pred_tail_type = tail[2]
                    pred_rel_type = self._input_reader.get_relation_type(rel.item())
                    score = pred_score.item()

                    head_start, head_end = head[0], head[1]
                    tail_start, tail_end = tail[0], tail[1]
                    converted_rel = ((head_start, head_end, pred_head_type),
                                         (tail_start, tail_end, pred_tail_type), pred_rel_type, score)
                    converted_rels.append(converted_rel)

        return converted_rels
    
    
    def _convert_pred_relations_upper(self, pred_types: torch.tensor, pred_scores: torch.tensor, 
                                pred_entities: List[tuple], token_mask: torch.tensor):
        converted_rels = []
        
        pred_types = torch.triu(pred_types, diagonal=1)
        for i,j in pred_types.nonzero():
            label_idx = pred_types[i,j].float()
            pred_rel_type = self._input_reader.get_relation_type(torch.ceil(label_idx/2).item())
            if label_idx % 2 == 1: # R-X
                head_idx = i 
                tail_idx = j 
            else: # L-X
                head_idx = j 
                tail_idx = i
            
            head_entity = self._find_entity(head_idx, token_mask, pred_entities)
            tail_entity = self._find_entity(tail_idx, token_mask, pred_entities)

            if head_entity == None or tail_entity == None:
                continue
            pred_head_type = head_entity[2]
            pred_tail_type = tail_entity[2]
            score = pred_scores[i][j].item()

            head_start, head_end = head_entity[0], head_entity[1]
            tail_start, tail_end = tail_entity[0], tail_entity[1]
            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type, score)

            converted_rels.append(converted_rel)
        return converted_rels
    
    def _get_token_span(self, entity, token_mask, encoding_length):
        subtoken_start = entity[0]
        subtoken_end = entity[1]
        token_span = []
        
        for i in range(1, encoding_length):
            if token_mask[i][subtoken_start:subtoken_end].any():
                token_span.append(i)
        return token_span
            
            
    def _find_entity(self, idx, token_mask, entities):
        # idx -> word level index
        span = token_mask[idx].nonzero().squeeze(0)
        for e in entities:
            if len(span) == 0:
                return None
            if span[-1] <= e[1] - 1 and e[0] - 1 <= span[-1]:
                return e
        return None


    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):

            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])
 
        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        gt_all = [gt for gt in gt_all]
        pred_all = [pred for pred in pred_all]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    

        
    
    def _dump_json(self, doc: Document, gt_entities: List[Tuple], pred_entities: List[Tuple],
                     gt_relations: List[Tuple], pred_relations: List[Tuple]):

        def entities_json(tokens, entities):
            ners = []
            ent2token = {}
            for e in entities:
                span = []
                for i,t in enumerate(tokens):
                    if subset_of(t,(e[0], e[1])):
                        span.append(i)
                ners.append([span[0], span[-1], e[2].short_name])
                ent2token[e[:2]] = ners[-1]
            return ent2token, ners
        
        def relations_json(tokens, relations, ent2token):
            rels = []
            for r in relations:
                rels.append(ent2token[r[0][:2]][:2] + ent2token[r[1][:2]][:2] + [r[2].short_name])
            return rels
        
        contents = {"sentences": [[t.phrase for t in doc._tokens]], "ner": [], "relations": [],
             "predicted_ner": [], "predicted_relations": [],
             "clusters": [], "doc_key": doc.doc_id}
        
        token_spans = [t.span for t in doc._tokens]
        
        gt_ent2token, gt_ners = entities_json(token_spans, gt_entities)
        contents["ner"].append(gt_ners)
#         print(token_spans, pred_entities)
        pred_ent2token, pred_ners = entities_json(token_spans, pred_entities)
        contents["predicted_ner"].append(pred_ners)
        contents["relations"].append(relations_json(token_spans, gt_relations, gt_ent2token))
        contents["predicted_relations"].append(relations_json(token_spans, pred_relations, pred_ent2token))
        
        with open('prediction.json', 'a+') as f:
            json.dump(contents, f)
            f.write('\n')
        
        
        return 
    
    
    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        encoding = doc.encoding
    
        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)

        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name
            # print("vvvv:", type_verbose)
            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                     + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
