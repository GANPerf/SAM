# Improving Fine-Grained Visual Recognition in Low Data Regimes via Self-Boosting Attention Mechanism 
#ECCV2022 Submission Paper ID 6680



## Dependencies
* python3.6
* torch == 1.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.4.2
* tensorboardX
* numpy
* argparse

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | http://www.vision.caltech.edu/visipedia/CUB-200-2011.html |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |



## RUN
- The running commands for several datasets are shown below. Please refer to ``run.sh`` for commands for datasets with other label ratios and label category.
```
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 10 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 10 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 10 --pretrained

```

