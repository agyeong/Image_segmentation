# Semantic Image Segmentation

## Data
- Original data : file01.png, file02.png, ...
- Labeled data : file01_L.png, file02\_L.png

## Class
- Total 11 class + Void class
- Pole, SignSymbol, Bicyclist, Pedestrian, Building, Fence, Pavement, Road, Car, Sky, Tree
- Class Distrubution  
  <img src="https://user-images.githubusercontent.com/70522267/121540609-95cf2e00-ca41-11eb-9d26-863c34fe34f2.png" width="400px" />

## Model
- DeepLab v3 resnet 50

## file
- run.py : Files that you run.
- model.py : Define Model Structure.
- labels.py : About pixel and label values in data.
- helpers.py : For calculation of iou, acc, loss, etc.

## Test set result
- epoch 100, batch 2
```
classes           IoU
---------------------
Pole          : 0.671
SignSymbol    : 0.872
Bicyclist     : 0.896
Pedestrian    : 0.829
Building      : 0.959
Fence         : 0.921
Pavement      : 0.949
Road          : 0.987
Car           : 0.958
Sky           : 0.941
Tree          : 0.925
---------------------
Mean IoU      : 0.901
---------------------
loss : 0.1683 acc : 0.9686 miou : 0.9005
```

## Environments
- CPU : Intel Core i9-10900X (10 Core / 20 Thread)
- RAM : Samsung 2666 (94G)
- GPU : GeForce RTX 2080 Ti (11G)

## Member
- [김아경](https://github.com/EP00) : bzxz55@gmail.com
- [유소영](https://github.com/yooso0731) : yoosomail@gmail.com

## reference
- [hoya012](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch)
