import torch
import json
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
from transformers import 

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

unk_token = "[UNK]"
pad_token = "[PAD]"
    
def glove_init(scale = "6B"):
    
    unk_index = 0
    pad_index = 1

    glove_vectors = GloVe(scale)
    glove_vocab = vocab(glove_vectors.stoi)
    glove_vocab.insert_token(unk_token, unk_index)
    glove_vocab.set_default_index(unk_index)
    glove_vocab.insert_token(pad_token, pad_index)


    with open("glove/glove_vocab.json", 'w') as fp:
        json.dump(glove_vocab.get_stoi(), fp)    

    pretrained_embeddings = glove_vectors.vectors
    pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))

    return glove_vocab, pretrained_embeddings

# vocab, emb = glove_init()

# index2vocab = vocab.get_itos()
# with open("glove/vocab.txt", 'w') as fp:
#     for word in index2vocab:
#         fp.write(word)
#         fp.write('\n')

    