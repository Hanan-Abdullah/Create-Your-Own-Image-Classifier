import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

def load_checkpoint(hidden_units= 4096 , arch = "vgg16", classes_num = 102  ):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('The Architecture Is Not Recognized', arch)
        
        
    for param in model.parameters():
        param.requires_grad = False
        
    if arch == 'vgg16':
        F_inputs = model.classifier[0].in_features
    else:
        F_inputs = model.classifier[1].in_features
    
    
    model.classifier = nn.Sequential(nn.Linear(F_inputs, hidden_units),
                                     nn.ReLU(),
                                     nn.Linear(hidden_units, int(hidden_units/4 )),
                                     nn.ReLU(),
                                     nn.Dropout(p = 0.2),
                                     nn.Linear(int(hidden_units/4) ,classes_num ),
                                     nn.LogSoftmax(dim = 1))
    
    return model


