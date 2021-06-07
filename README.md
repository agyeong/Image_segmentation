## DLv3 

### How to use    

1. Prepare using dataset    
``` 
  {PROJECT_DIR}/data/Original_data   
  {PROFECT_DIR}/data/Labeled_data     // annatated dataset. *_L.png   
```  
2. Data preprocessing   
  주피터 환경에서 **data_preprocessing.ipynb** 실행    
  *npz file:: train/val/test (0.7, 0.1, 0.2 %) 분할 파일   
  --> data_origin.npz: 원본 데이터    
  --> data_norm_v1.npz: 입력 데이터({}_x) 정규화 ver1   
  --> data_norm_v2.npz: 입력 데이터 정규화 ver2 + color jitter 처리한 데이터     
  
3. RUN training file (*.ipynb)   
    >  **RUN_weight**: class weight만 적용한 버전   
    >  **RUN_Adam**: weight + 최적화 알고리즘 SGD -> Adam 변경    
    >  **RUN_norm**: class weight + norm ver1   
    >  **RUN_norm2**: weight + norm ver2   
    >  **RUN_jitter**: weight + color jitter   
    >  **RUN_new**: change model architecture (plus 1 hidden layer in decoding part)   
