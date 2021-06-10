# load packages 
import os
import sys
import time
import datetime
from tqdm import tqdm
import numpy as np
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd
import matplotlib.pyplot as plt

from model import DeepLabHead
from helpers import AverageMeter, ProgressMeter, iouCalc
from labels import labels

# data path
origin_data_path = os.getcwd() + '/data/Original_data/'
labeled_data_path = os.getcwd() + '/data/Labeled_data/'

origin_data_list = os.listdir(origin_data_path) # x
labeled_data_list = os.listdir(labeled_data_path) # y

# file random shuffle
random.shuffle(origin_data_list)
random.shuffle(labeled_data_list)

# split data (train, test / x, y) 
train_x_file = origin_data_list[:int(len(origin_data_list)*0.8)]
train_y_file = [file_name[:-4] + '_L.png' for file_name in train_x_file]
test_x_file = [file_name for file_name in origin_data_list if file_name not in train_x_file]
test_y_file = [file_name[:-4] + '_L.png' for file_name in test_x_file]

# X
train_x = [np.array(Image.open(origin_data_path + train)) for train in train_x_file]
test_x = [np.array(Image.open(origin_data_path + test)) for test in test_x_file]

# Y
# color to label catId
color2label = { label.color   : label.id for label in labels}

print('Start pixel to label id - Train data set')
print('cf. Expected 20 minutes')
train_y = []
for file_name in tqdm(train_y_file):
    image = np.array(Image.open(labeled_data_path + file_name))
    ret = [[color2label[tuple([r[0], r[1], r[2]])] 
            if tuple([r[0], r[1], r[2]]) in color2label else 11
            for r in row] 
           for row in image]
    train_y.append(ret)
print('End pixel to label id - Train data set')

print('Start pixel to label id - Test data set')
print('cf. Expected 5 minutes')
test_y = []
for file_name in tqdm(test_y_file):
    image = np.array(Image.open(labeled_data_path + file_name))
    ret = [[color2label[tuple([r[0], r[1], r[2]])] 
            if tuple([r[0], r[1], r[2]]) in color2label else 11
            for r in row] 
           for row in image]
    test_y.append(ret)
print('End pixel to label id - Test data set')
    
# make data set for model training 
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
    
print("========================= model =========================")

# choose CPU or GPU
USE_CUDA = torch.cuda.is_available() and True # True -> GPU / False -> CPU
device = torch.device('cuda' if USE_CUDA else 'cpu')

# model load
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False).to(device)
model.classifier = DeepLabHead(2048, 12).to(device) # 12 = class num

# parameter setting
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2) # setting parameter learning rate 
    
# Create list of class names
classLabels = [] # class name ex) Pole, SignSymbol, Bicyclist...
for label in labels:
    if label.name not in classLabels:
        classLabels.append(label.name)
classLabels.append('void') 
validClasses = list(np.unique([label.id for label in labels if label.id >= 0] + [11])) # class number ex) 0, 1, 2, ...


print("========================= Train Start =========================")
print('cf. Expected 5 minutes per 1 epoch')
num_epoch = 200
batch = 2
res = train_X.shape[1] * train_X.shape[2]

train_X = torch.tensor(train_x, dtype=torch.float32)
train_Y = torch.tensor(train_y, dtype=torch.long)

train_data = torch.utils.data.TensorDataset(train_X.permute(dims=(0, 3, 1, 2)), train_Y)
train_data = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)   

# class weight 
dist = {i:(train_Y == i).sum().tolist() for i in range(12)}

weights = [1/dist[i] for i in range(12)]
total_weights = sum(weights)

class_weight = torch.FloatTensor([w/total_weights for w in weights]).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weight, ignore_index=12) 

for epoch in range(num_epoch):
    model.train()

    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')  
    iou = iouCalc(classLabels, validClasses, voidClass = 11)
    progress = ProgressMeter(
        len(train_data),
        [loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    for batch, (x, y) in enumerate(tqdm(train_data, total=len(train_data))):
        
        x = x.to(device)
        y = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward passQD
        outputs = model(x)
        outputs = outputs['out']
        preds = torch.argmax(outputs, 1)
        
        # cross-entropy loss
        loss = criterion(outputs, y)

        # backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        bs = x.size(0)
        loss = loss.item()
        loss_running.update(loss, bs)
        corrects = torch.sum((preds == y) & (y != 12))
        
        nvoid = int((y==12).sum())
        acc = corrects.double()/(bs*res-nvoid)
        acc_running.update(acc, bs)
        
        # Calculate IoU scores of current batch
        iou.evaluateBatch(preds, y)
        
    scheduler.step(loss_running.avg)
    miou = iou.outputScores()
    
    print('train epoch ', epoch+1)
    print('loss : {:.4f}   acc : {:.4f}   miou : {:.4f}'.format(loss_running.avg, acc_running.avg, miou))
    
#     # save checkpoint per epoch
#     now = datetime.datetime.now()
#     now_time = now.strftime('%y%m%d_%H:%M')
    
#     # save checkpoint per epoch
#     now = datetime.datetime.now()
#     now_time = now.strftime('%y%m%d_%H:%M')
      
    # save path
    if not os.path.isdir(os.getcwd() + '/result/'):
        os.makedirs(os.getcwd() + '/result/')
    
    save_path = os.getcwd() + '/result/'
    
#     with open(save_path + 'train_log_epoch.csv', 'a') as epoch_log:
#             epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(
#                     epoch+1, loss_running.avg, acc_running.avg, miou))
    
#     # Save best model to file
#     torch.save({
#         'epoch' : epoch+1,
#         'model_state_dict' : model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'best_miou': best_miou,
#         'metrics': metrics,
#         }, save_path + now_time + '_checkpoint.pth.tar')
    
    # Save best model to file
    if miou > best_miou:
        print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou))
        best_miou = miou
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            }, save_path + 'best_weights.pth.tar')


print("========================= Train Start =========================")
X = torch.tensor(test_x, dtype=torch.float32)
Y = torch.tensor(test_y, dtype=torch.long)

data = torch.utils.data.TensorDataset(X.permute(dims=(0, 3, 1, 2)), Y)

test_data = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

# Load best model
save_path = os.getcwd() + '/result/'
bestmodel = [file for file in sorted(os.listdir(save_path), reverse=True) if file == 'best_weights.pth.tar' ][0]
checkpoint = torch.load(save_path + bestmodel) 
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print('Loaded best model weights (epoch {}) from {}'.format(checkpoint['epoch'], save_path + bestmodel))

model.eval()

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
progress = ProgressMeter(
    len(test_data),
    [batch_time, data_time],
    prefix='Predict: ')

for batch, (x, y) in enumerate(tqdm(test_data, total=len(test_data))):

    x = x.to(device)
    y = y.to(device)

    # forward
    outputs = model(x)
    outputs = outputs['out']

    preds = torch.argmax(outputs, 1)

    # cross-entropy loss
    loss = criterion(outputs, y)

    # Statistics
    bs = x.size(0)
    loss = loss.item()
    loss_running.update(loss, bs)
    corrects = torch.sum((preds == y) & (y != 12))

    nvoid = int((y==12).sum())
    acc = corrects.double()/(bs*res-nvoid)
    acc_running.update(acc, bs)

    # Calculate IoU scores of current batch
    iou.evaluateBatch(preds, y)

miou = iou.outputScores()
scheduler.step(loss_running.avg)

print('loss : {:.4f} acc : {:.4f} miou : {:.4f}'.format(loss_running.avg, acc_running.avg, miou))






    
    
    
    
    
    
    