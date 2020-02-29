import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
import argparse 
from Load_Checkpoint import load_checkpoint
from utiliy_functions import process_image
from PIL import Image
import json

parser = argparse.ArgumentParser()

parser.add_argument('--ImagePath', type = str, help = 'Path of an Image')
parser.add_argument('--top_k', type = int, help = 'top K most likely classes')
parser.add_argument('--category_names', type = str, help = 'mapping of categories to real names')
parser.add_argument('--mode', type = str, help = 'mode')



args = parser.parse_args()

if args.ImagePath:
    ImagePath = args.ImagePath
    
    
if args.top_k:
    top_k = args.top_k
    
    
if args.category_names:
    category_names = args.category_names
    
    
if args.mode:
    mode = args.mode
    
checkpoint = torch.load ('checkpoint.pth')   
hidden_units = checkpoint['hidden_units']
arch = checkpoint['arch']

model = load_checkpoint(hidden_units, arch, 102)

state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.class_to_idx = checkpoint['class_to_idx']


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device(mode) 
    model.to(device)
    
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image_inputs = torch.unsqueeze(image_tensor,0)
    
    image_inputs = image_inputs.to(device)
    
    log_ps = model(image_inputs)
    ps = torch.exp(log_ps)
    
    top_p, top_classes = ps.topk(topk, dim = 1)

    idx_to_class = dict((value,key) for key,value in model.class_to_idx.items())
    tops_classes_1 = top_classes.detach().cpu().numpy().tolist()[0] 
    tops_probs = top_p.detach().cpu().numpy().tolist()[0] 
   
    tops_classes_2 = list()
    flowers = []
    
    for i in tops_classes_1:
        tops_classes_2.append(idx_to_class[i])
        
        
    for i in range(len(tops_classes_2)):
        flowers.append(cat_to_name[str(tops_classes_2[i])].title())
        

    for idx in range(len(tops_probs)):
        print("{}- {}: {}".format(idx,flowers[idx], tops_probs[idx]))
        
    return tops_probs, tops_classes_2, flowers


probs, classes, flowers_names  = predict(ImagePath, model, top_k)


       

