import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, AutoConfig, BertPreTrainedModel


from src import sampling
from src import util


from src.entities import Token
from src.attention import TableTransformer, TableTransformerLayer
from src.beam import Beam


from typing import List
from pytorch_memlab import profile, MemReporter
import math

class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=512, dropout=0.):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)   
        self.fc2 = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ConvNet(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim, kernel_size=3, stride=1, padding='same', dropout=0.3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hid_dim, kernel_size, stride, padding)   
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(hid_dim, output_dim, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
#         x = F.relu(self.conv2(x))
#         x = self.dropout(x)
        return self.conv3(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
       

class _2DTrans(BertPreTrainedModel):
    
    """ The table filling model for jointly extracting entities and relations using pretrained BERT.
    
    Params:
    :config: configuration for pretrained BERT;
    :relation_labels: number of relation labels;
    :entity_labels: number of entity labels;
    :entity_label_embedding: dimension of NE label embedding;
    :rel_label_embedding: dimension of hidden attention for relation classification;
    :prop_drop: dropout rate;
    :freeze_transformer: fix transformer parameters or not;
    :device: devices to run the model at, e.g. "cuda:1" or "cpu".
    
    """

    def __init__(self, config: AutoConfig, entity_labels: int, relation_labels: int,
                 entity_label_embedding: int,  rel_label_embedding: int,
                 pos_embedding: int, encoder_embedding: int, encoder_hidden: int,
                 encoder_heads: int, encoder_layers: int, attn_type: str,
                 prop_drop: float, freeze_transformer: bool, device):
        
        super(_2DTrans, self).__init__(config)
        # BERT model
        self.bert = AutoModel.from_config(config)
        
        self._device = device
        self._attn_type = attn_type
        
        # embeddings for entity
        self.entity_label_embedding = nn.Embedding(entity_labels, entity_label_embedding)
        
        # embeddings for relation
#         self.rel_label_embedding = nn.Embedding(relation_labels, rel_label_embedding)
        
        # 2d-transformer
#         self.pos_embedding = nn.Embedding(self._context_size * 2, pos_embedding)
#         encoder_dim = (config.hidden_size + entity_label_embedding) * 2 + rel_label_embedding
        encoder_dim = config.hidden_size * 2 
#         self.mlp = MLPNet(encoder_dim, encoder_embedding)
        
#         self.pos_encoder = PositionalEncoding(d_model = config.hidden_size, dropout = prop_drop)
        
#         self.encoder_layers = TableTransformerLayer(n_heads = encoder_heads, input_dim = encoder_dim,  hid_dim = encoder_hidden, attn_type = attn_type, device = device, activation="relu", dropout=prop_drop)
            
#         self.encoder = TableTransformer(self.encoder_layers, encoder_layers)
#         self.encoder = MLPNet(encoder_dim, encoder_dim)
        self.encoder = ConvNet(encoder_dim, encoder_hidden, relation_labels)

        self.ent_classifier = nn.Linear(relation_labels, entity_labels)
#         self.rel_classifier = nn.Linear(encoder_hidden, relation_labels)
        
        self.dropout = nn.Dropout(prop_drop)
        
        self._entity_labels = entity_labels
        self._relation_labels = relation_labels

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze encoder weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False



    def _forward_table(self, h: torch.tensor, token_context_masks: torch.tensor,
                       entity_masks: torch.tensor, entity_preds: torch.tensor, rel_preds: torch.tensor):
        
        entity_labels = entity_preds
        # wipe out previous predictions
#         entity_labels = torch.zeros_like(entity_preds)
#         rel_preds = torch.zeros_like(rel_preds)
    
        # wipe out word embeddings
#         h = torch.zeros_like(h)
        
        batch_size, context_size = entity_labels.shape
        # entity span repr.
        
#         entity_masks = util.extend_tensor(entity_masks, token_masks.shape)
#         entity_repr_pool = util.max_pooling(h, entity_masks)

        # entity_label span repr.
#         entity_label_embeddings = self.entity_label_embedding(entity_labels)  
#         entity_label_pool = util.max_pooling(entity_label_embeddings, entity_masks)

        # rel repr for classification.
        
#         embed_rel = self.rel_label_embedding(rel_preds)
    
#         entry_repr = torch.cat([h, entity_label_embeddings], dim=2)
        entry_repr = h
        entry_repr[entry_repr == -1e30] = 0
#         entry_repr = self.pos_encoder(entry_repr)
#         entry_repr = torch.cat([entity_repr_pool, entity_label_pool], dim=2)
        entry_repr = entry_repr.unsqueeze(1).repeat(1, h.shape[1], 1, 1)
        
        
#         rel_repr = torch.cat([entry_repr.transpose(1,2), entry_repr, embed_rel], dim=3)
        rel_repr = torch.cat([entry_repr.transpose(1,2), entry_repr], dim=3)
#         print("rel repr:", rel_repr)
        encoder_repr = self.dropout(rel_repr)

        attention = self.encoder(encoder_repr.permute(0,3,1,2))


        ent_logits = self.ent_classifier(attention.diagonal(dim1=2,dim2=3).transpose(2,1))
        
#         rel_logits = self.rel_classifier(attention.permute(0,2,3,1))
        rel_logits = attention.permute(0,2,3,1)
#         print(ent_logits.shape, rel_logits.shape)
        return ent_logits, rel_logits


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, 
                        token_masks: torch.tensor, token_context_masks: torch.tensor,
                       entity_masks: torch.tensor, bert_layer: int,
                       pred_entities: torch.tensor, pred_relations: torch.tensor):  
        
        ''' Forward step for training.
        
        Params:
        :encodings: token encodings (in subword), of shape (batch_size, subword_sequence_length);
        :context_mask: masking out [PAD] from encodings, of shape (batch_size, subword_squence_length);
        :token_mask: a tensor mapping subword to word (token), of shape (batch_size, n+2, subword_sequence_length);
        :gold_entity: ground-truth sequence for NE labels, of shape (batch_size, n+2);
        :entity_masks: ground-truth mask for NE spans, of shape (batch_size, n+2, n+2);
        :gold_rel: ground-truth matrices for relation labels, of shape (batch_size, n+2, n+2);        
        :allow_rel: whether allow re predictions or not.
        
        Return:
        
        :all_entity_logits: NE scores for each word on each batch, a list of length=batch_size containing tensors of shape (1, n, entity_labels);
        :all_rel_logits: relation scores for each word pair on each batch, a list of length=batch_size containing tensors of shape (1, relation_labels, n, n).
        
        '''
        
        # get contextualized token embeddings from last transformer layer
        outputs = self.bert(input_ids=encodings, attention_mask=context_masks.float())
        last_hidden = outputs[0]
        h = outputs[-1][bert_layer]
        
        token_spans_pool = util.max_pooling(h, token_masks)

        entity_logits, rel_logits = self._forward_table(token_spans_pool, token_context_masks, entity_masks, pred_entities, pred_relations)

#         for batch in range(batch_size): # every batch
            
#             batch_h = h[batch]
#             token_mask = token_masks[batch]
#             entity_mask = entity_masks[batch]
#             pred_ent = pred_entities[batch]
#             pred_rel = pred_relations[batch]      

#             num_steps = pred_ent.shape[-1]
        
#             # map from subword repr to word repr.

#             word_h = batch_h * token_mask.unsqueeze(-1)
#             word_h_pooled = word_h.max(dim=1)[0]

#             word_h_pooled = word_h_pooled[:num_steps+2].contiguous()
#             word_h_pooled[0,:] = 0
#             # curr word repr.
#             curr_word_repr = word_h_pooled[1:-1].contiguous()

#             # unsqueeze the first dimension to match (beam_size, n+2, n+2) for evaluation
#             print("batch:", batch)
#             curr_ent_logits, curr_rel_logits = self._forward_table(curr_word_repr, entity_mask.unsqueeze(0), pred_ent, pred_rel)
#             all_entity_logits.append(curr_ent_logits)
#             all_rel_logits.append(curr_rel_logits)

        return entity_logits, rel_logits

    
    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, 
                        token_masks: torch.tensor, token_context_masks: torch.tensor,
                       entity_masks: torch.tensor, bert_layer: int,
                       pred_entities: torch.tensor, pred_relations: torch.tensor):   
        
        return self._forward_train(encodings, context_masks, token_masks, token_context_masks, entity_masks, bert_layer, pred_entities, pred_relations)


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)
        
        
# Model access

_MODELS = {
    '2d_trans': _2DTrans,
    }

def get_model(name):
    return _MODELS[name]
