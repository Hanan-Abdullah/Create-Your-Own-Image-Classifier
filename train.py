import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse 
from Load_Checkpoint import load_checkpoint
from utiliy_functions import loadData
from workspace_utils import keep_awake


parser = argparse.ArgumentParser()
         
parser.add_argument('--data_dir', type = str, help = 'dataset path')
parser.add_argument('--learning_rate', type = float, help = 'learning_rate')
parser.add_argument('--hidden_units', type = int, help = 'Number of hidden_units')
parser.add_argument('--epochs', type = int, help = 'Number of epochs')
parser.add_argument('--arch', type = str, help = 'Model Architecture')
parser.add_argument('--checkpoints', type = str, help = 'save traind model int checkpoint file')
parser.add_argument('--mode', type = str, help = ' mode')



args = parser.parse_args()

if args.data_dir:
    data_dir = args.data_dir
    
if args.learning_rate:
    learning_rate = args.learning_rate
        
if args.hidden_units:
    hidden_units = args.hidden_units
    
if args.epochs:
    epochs = args.epochs 
    
if args.arch:
    arch = args.arch 
        
if args.checkpoints:
    checkpoints = args.checkpoints
    
if args.mode:
    mode = args.mode



def train_model(arch = "vgg16", hidden_units = 4069, checkpoint = " ", epochs = 3, learning_rate = 0.003):
    for i in keep_awake(range(1)):
        device = torch.device(mode)
        
        dataset, dataloader = loadData(data_dir) 
        trainloader = dataloader['train']
        train_dataset = dataset['train']
        validloader = dataloader['valid']
        valid_dataset = dataset['valid']

        classes_num = len(train_dataset.classes)
    
        model = load_checkpoint(hidden_units , arch , classes_num)
    
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
        criterian = nn.NLLLoss()
    
        model.to(device)
        epochs_num = epochs
        step = 0
        running_loss = 0
        print_every = 5
    
        for epoch in range (epochs_num):
            for images,labels in trainloader:
                step += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(images)
                loss = criterian(logps,labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                if step % print_every == 0:
                    vaild_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        
                        #Validation Loop
                        for images,labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            logps = model(images)
                            batch_loss = criterian(logps, labels)
                            vaild_loss += batch_loss.item()
                    
                            #Claculate The Accuracy
                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1,dim = 1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.."
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Validation loss: {vaild_loss/len(validloader):.3f}.. "
                              f"Validation accuracy: {accuracy/len(validloader):.3f}")
                        running_loss = 0
                        model.train()
                        
        model.class_to_idx = train_dataset.class_to_idx
        checkpoint_dictionary = {'hidden_units': hidden_units,
                                 'arch':arch,
                                 'class_to_idx': model.class_to_idx,
                                 'state_dict':model.state_dict()}
        torch.save(checkpoint_dictionary, checkpoint)
    return model 


train_model(arch , hidden_units , checkpoints , epochs , learning_rate )

