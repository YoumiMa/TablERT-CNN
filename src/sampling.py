import random

import torch

from src import util


def collect_entities(entities, context_size, token_count):
    # positive entities
    ent_labels = torch.zeros(context_size, dtype=torch.long)
    
    ent_masks = torch.zeros((context_size, context_size), dtype=torch.bool)
    ent_masks.fill_diagonal_(True)
    
    for e in entities:
#         print("phrase:", e.phrase, "tokens", e.tokens, len(e.tokens))
        ent_mask = create_ent_mask(*e.span, context_size)
        ent_masks[e.span[0]:e.span[1], e.span[0]:e.span[1]] = True
#         ent_mask[*e.span] = True
        for i, t in enumerate(e.tokens):
            ent_labels[t.index + 1] = e.entity_labels[i].index
    
    
    return ent_labels, ent_masks

def collect_rels(rels, context_size):
    # positive relations

    rel_labels = torch.zeros((context_size, context_size), dtype=torch.long)

    for rel in rels:
        # relation_labels[rel.tail_ent.span]
        head = rel.head_entity
        tail = rel.tail_entity

        # ## map to all words in an ent.
        for i in range(head.span_word[0], head.span_word[1]):
            for j in range(tail.span_word[0], tail.span_word[1]):
                rel_labels[i + 1][j + 1] = rel.relation_type.index * 2 - 1
                rel_labels[j + 1][i + 1] = rel.relation_type.index * 2                 

        ### map to last word in an ent.
        # for i in range(former._tokens[-1].span_start, former._tokens[-1].span_end):
        #     for j in range(latter._tokens[-1].span_start, latter._tokens[-1].span_end):
        #         rel_labels[i][j] = rel.rel_label.index    
    return rel_labels


def create_sample(doc, shuffle = False):
    
    encoding = doc.encoding

#     print(doc.doc_id)
    context_size = len(encoding)
    token_count = len(doc.tokens)

    ent_labels, ent_masks = collect_entities(doc.entities, context_size, token_count)
#     print("entity labels:", ent_labels)
#     print("masks:", ent_masks)
    
    rel_labels = collect_rels(doc.relations, context_size)
    
    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # context mask
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # token masks
    tokens = doc.tokens
#     token_masks = torch.zeros((context_size, context_size), dtype=torch.bool)
    token_masks = torch.zeros((len(_encoding), context_size), dtype=torch.bool)
    token_ctx_mask = torch.zeros(context_size, dtype=torch.bool)

    for i,t in enumerate(tokens):
        token_masks[i+1, t.span_start:t.span_end] = 1
#         print(t.span_start, t.span_end)
        token_ctx_mask[i + 1] = 1

    
    torch.set_printoptions(edgeitems=15)

#     print("entity mask:", ent_masks, ent_masks.shape)
    return dict(encodings=encoding, ctx_masks=ctx_mask, ent_masks=ent_masks,
                            ent_labels=ent_labels, rel_labels=rel_labels, 
                            token_masks=token_masks, token_ctx_masks=token_ctx_mask)



def create_ent_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_ent_mask(start, end, context_size)
    return mask

def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch