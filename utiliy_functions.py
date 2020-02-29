import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse 
from PIL import Image


def loadData (data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

 
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    train_dataset = datasets.ImageFolder(train_dir,transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir,transform = valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset,batch_size = 64, shuffle = True) 
    
    dataset = {'train': train_dataset,
               'valid': valid_dataset}
    
    dataloader = {'train': trainloader,
                  'valid': validloader}
    
    return dataset, dataloader


def process_image(ImagePath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(ImagePath)
    
    #1: Resize the image
    width, height = im.size  
    if width < height:
        im.thumbnail((256,256 *(width / height)))
    else:
        im.thumbnail((256 *(width / height),256))
        
    #2: Crop the image 
    w, h = im.size
    left = (w - 224) / 2
    top =  (h - 224) / 2
    right = (w + 224) / 2
    bottom = (h + 224) / 2
    cropped_img = im.crop((left, top, right, bottom))
    
    #3: Color channels of the image
    np_image = np.array(cropped_img)/255
    
    #4: Normlized Image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([[0.229, 0.224, 0.225]])
    img = (np_image - mean) / std
    
    #5: Reorder dimensions
    img = img.transpose(2,0,1)
    
    return img


