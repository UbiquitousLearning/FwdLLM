from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)
import torch.nn as nn
import torch
import functorch as fc
from functools import partial
from torch.nn import CrossEntropyLoss

def _get_loss(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(x.view(-1, num_classes), t.view(-1))
    return loss

def functional_get_loss(
    params,
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int = 10,
    buffers = None
) -> torch.Tensor:

    y = model(params,buffers, x)[0]
    return _get_loss(y, t, num_classes)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model.add_module('pre_classifier',nn.Sequential())
model.add_adapter("rotten tomato")
model.train_adapter("rotten tomato")

x = torch.randint(256,(4,256))
labels = torch.tensor([0,1,1,1])

fmodel, params, buffers = fc.make_functional_with_buffers(model)

# v_params = tuple([torch.randn_like(p) if p.requires_grad == True else torch.zeros_like(p) for p in params])
v_params = tuple([torch.randn_like(p) for p in params])

f = partial(
            functional_get_loss,
            model=fmodel,
            # model=partial(fmodel,buffers=buffer),
            buffers = buffers,
            num_classes = 2,
            x=x,
            t=labels,
        )

loss, jvp = fc.jvp(f, (params,), (v_params,))