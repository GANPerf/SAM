
## CUB200
## 200 category 
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 10 --pretrained 
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 30 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 50 --pretrained

## 100 category 
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 10200100 --pretrained --class_num 100
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 15200100 --pretrained --class_num 100
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 30200100 --pretrained --class_num 100
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 50200100 --pretrained --class_num 100

## 50 category 
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1020050 --pretrained --class_num 50
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1520050 --pretrained --class_num 50
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 3020050 --pretrained --class_num 50
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 5020050 --pretrained --class_num 50

## 30 category 
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1020030 --pretrained --class_num 30
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1520030 --pretrained --class_num 30
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 3020030 --pretrained --class_num 30
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 5020030 --pretrained --class_num 30


## StanfordCars
## 196 category 
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 10 --pretrained
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 15 --pretrained
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50  --label_ratio 30 --pretrained
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50  --label_ratio 50 --pretrained

## 100 category 
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 10196100 --pretrained --class_num 100
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 15196100 --pretrained --class_num 100
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50  --label_ratio 30196100 --pretrained --class_num 100
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50  --label_ratio 50196100 --pretrained --class_num 100

## 50 category 
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 1019650 --pretrained --class_num 50
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 1519650 --pretrained --class_num 50
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50  --label_ratio 3019650 --pretrained --class_num 50
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50  --label_ratio 5019650 --pretrained --class_num 50

## 30 category 
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 1019630 --pretrained --class_num 30
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50  --label_ratio 1519630 --pretrained --class_num 30
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50  --label_ratio 3019630 --pretrained --class_num 30
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50  --label_ratio 5019630 --pretrained --class_num 30

## Aircraft
## 100 category
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 10 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 30 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 50 --pretrained

## 50 category
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1010050 --pretrained --class_num 50
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1510050 --pretrained --class_num 50
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 3010050 --pretrained --class_num 50
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 5010050 --pretrained --class_num 50

## 30 category
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1010030 --pretrained --class_num 30
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --backbone resnet50 --label_ratio 1510030 --pretrained --class_num 30
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 1 --backbone resnet50 --label_ratio 3010030 --pretrained --class_num 30
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --backbone resnet50 --label_ratio 5010030 --pretrained --class_num 30





