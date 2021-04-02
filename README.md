# ChannelDropBlock_Pytorch

### Dependencies:
Python 3.6vwith all of the `pip install -r requirements.txt` packages including:
- `torch == 1.3.1`
- `opencv-python`

### Data
1. Download the FGVC image data. Extract them to `data/cars/`, `data/birds/` and `data/airs/`, respectively. Organize the structure as follows:
```
dataset/
    └── train/
         └── class1/
              └── img1.jpg
              └── img2.jpg
              └── ...
         └── ...
     └── test/
         └── class1/
              └── img1.jpg
              └── img2.jpg
              └── ...
         └── ...
```

### Training:
1. For the CUB-200-2011 dataset, run `python train_birds+.py --model {resnet50,vgg19} --cdb {none,max_activation,bilinear_pooling} [options: --visualize]` to start training.
2. For the Stanford-Cars dataset, run `python train_cars.py --model {resnet50,vgg19} --cdb {none,max_activation,bilinear_pooling} [options: --visualize]` to start training.
3. For the  FGVC-Aircraft dataset, run `python train_airs.py --model {resnet50,vgg19} --cdb {none,max_activation,bilinear_pooling} [options: --visualize]` to start training.
4. Run `python {train_birds+.py, train_cars.py, train_airs.py} --help` to see full input arguments.

**Visualize:** 
1. Visualize online attention dropped/remianed feature maps under folder `visual/`.