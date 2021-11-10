from abc import ABC

import torch
import torch.nn.functional as F
from pytorch_memlab import profile

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class TableLoss(Loss):
    """ Compute the loss for training joint entity and relation extraction."""
    
    def __init__(self, rel_criterion, entity_criterion, model, optimizer = None, scheduler = None, max_grad_norm = None):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
#         self._device = model.output_device
        self._device = model._device
    
#     @profile
    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels, ctx_masks, is_eval=False):
        
        
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_labels = entity_labels.view(-1)

#         entity_masks = entity_masks.view(-1).float()
        entity_loss = self._entity_criterion(entity_logits, entity_labels)
        
#         print(rel_labels.shape, rel_logits.shape)
        rel_loss = self._rel_criterion(rel_logits.permute(0,3,1,2), rel_labels)
#         print((ctx_masks.unsqueeze(1) * ctx_masks.unsqueeze(1).permute(0,2,1)).long())

#         print("pre rel:", rel_logits.argmax(dim=-1) * (ctx_masks.unsqueeze(1) * ctx_masks.unsqueeze(1).permute(0,2,1)))
        # no diagonal
        rel_loss = rel_loss * (~torch.eye(rel_loss.shape[-1], dtype=torch.bool, device=self._device))
         
        entity_loss = entity_loss[ctx_masks.view(-1)].sum()
        # context mask
        rel_loss = rel_loss[ctx_masks.unsqueeze(1) * ctx_masks.unsqueeze(1).permute(0,2,1)].sum()

        train_loss = entity_loss + rel_loss

        loss = torch.tensor([entity_loss.item(), rel_loss.item(), train_loss.item()])
#         print("loss:", loss)
        if not is_eval:
#             print(torch.cuda.memory_summary())
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            torch.cuda.empty_cache()
            self._model.zero_grad()
        return loss