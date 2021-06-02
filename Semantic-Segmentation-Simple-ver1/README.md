# DeepLap v3 
### Description
- semantic image segmentation
- ref : https://github.com/hoya012/semantic-segmentation-tutorial-pytorch
### Files
- helpers.py : For calculation of iou, acc, loss, etc.
- labels.py : About pixel and label values in data.
- model.py : Define Model Structure.
- RUN.ipynb : Files that you run. (Original version)
- Class_Distribution.ipynb : class distribution for class weight.
- RUN_01_resnet101.ipynb : using CPU, pretrain False
- RUN_02_resnet50.ipynb (our SOTA model) : using GPU, pretrain False, add class weight, add validation code (but not run because of memory error) 

### How to use
Create a file called Data, then insert the data you want to use. 
```
data/Labeled_data
data/Original_data
```
