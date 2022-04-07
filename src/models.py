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
import math

class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=512, dropout=0.):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)   
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class ConvNet2layer(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim, kernel_size=3, stride=1, padding='same', dropout=0.3):
        super(ConvNet2layer, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hid_dim, kernel_size, stride, padding)   
        self.conv2 = nn.Conv2d(hid_dim, output_dim, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        return self.conv2(x)

class ConvNet3layer(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim, kernel_size=3, stride=1, padding='same', dropout=0.3):
        super(ConvNet3layer, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hid_dim, kernel_size, stride, padding)   
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(hid_dim, output_dim, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        return self.conv3(x)
    

class TablertCNN(BertPreTrainedModel):
    
    """ The model for jointly extracting entities and relations using pretrained BERT.
    
    Params:
    :config: configuration for pretrained BERT;
    :relation_labels: number of relation labels;
    :entity_labels: number of entity labels;
    :encoder_hidden: dimension of hidden states of CNN;
    :kernel_size: size of convolutional kernel; 
    :prop_drop: dropout rate;
    :freeze_transformer: fix transformer parameters or not;
    :device: devices to run the model at, e.g. "cuda:1" or "cpu".
    
    """

    def __init__(self, config: AutoConfig, entity_labels: int, relation_labels: int,
                 encoder_hidden: int, kernel_size: int, conv_layers: int,
                 prop_drop: float, freeze_transformer: bool, device):
        
        super(TablertCNN, self).__init__(config)
        # BERT model
        self.bert = AutoModel.from_config(config)
        
        self._device = device
        
        # Encoder
        encoder_dim = config.hidden_size * 2 

        if conv_layers == 2:
            self.encoder = ConvNet2layer(encoder_dim, encoder_hidden, relation_labels, kernel_size)
        elif conv_layers == 3:
            self.encoder = ConvNet3layer(encoder_dim, encoder_hidden, relation_labels, kernel_size)

        self.ent_classifier = nn.Linear(relation_labels, entity_labels)
        
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
        



    def _forward_table(self, h: torch.tensor):
        
        
        # entity span repr.
        entry_repr = h
        entry_repr[entry_repr == -1e30] = 0
        entry_repr = entry_repr.unsqueeze(1).repeat(1, h.shape[1], 1, 1)
        
        rel_repr = torch.cat([entry_repr.transpose(1,2), entry_repr], dim=3)
        encoder_repr = self.dropout(rel_repr)

        logits = self.encoder(encoder_repr.permute(0,3,1,2))
        ent_logits = self.ent_classifier(logits.diagonal(dim1=2,dim2=3).transpose(2,1))
        rel_logits = logits.permute(0,2,3,1)
    
        return ent_logits, rel_logits


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, 
                        token_masks: torch.tensor, bert_layer: int):  
        
        ''' Forward step for training.
        
        Params:
        :encodings: token encodings (in subword), of shape (batch_size, subword_sequence_length);
        :context_mask: masking out [PAD] from encodings, of shape (batch_size, subword_squence_length);
        :token_mask: a tensor mapping subword to word (token), of shape (batch_size, n+2, subword_sequence_length);
        :bert_layer: the layer index of BERT encoder whose outputs are used as sub-word embeddings;
        
        Return:
        
        :entity_logits: NE scores for each word on each batch, a list of length=batch_size containing tensors of shape (1, n, entity_labels);
        :rel_logits: relation scores for each word pair on each batch, a list of length=batch_size containing tensors of shape (1, relation_labels, n, n).
        
        '''
        
        # get contextualized token embeddings from last transformer layer
        outputs = self.bert(input_ids=encodings, attention_mask=context_masks.float())
        h = outputs[-1][bert_layer]
        
        token_spans_pool = util.max_pooling(h, token_masks)
        entity_logits, rel_logits = self._forward_table(token_spans_pool)


        return entity_logits, rel_logits

    
    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, 
                        token_masks: torch.tensor, bert_layer: int):   
        
        return self._forward_train(encodings, context_masks, token_masks, bert_layer)


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)
        
        
# Model access

_MODELS = {
    'tablert_cnn': TablertCNN,
    }

def get_model(name):
    return _MODELS[name]
