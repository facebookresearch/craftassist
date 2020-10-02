End-to-End Object Detection with Transformers
========

## Installation

Install PyTorch and torchvision
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Should be good to go.

## Train locally
For debugging purposes:
```
python detection.py
```

## Train on FAIR cluster
Install submitit:
```
pip install -U git+ssh://git@github.com/fairinternal/submitit@master#egg=submitit
```
Train baseline DETR-6-6 model on 4 nodes for 300 epochs:
```
python run_with_submitit.py --aux_loss --pass_pos_and_query --timeout 3000
```
