@echo off
cd C:\Users\Administrator\yolov5

python train.py --batch -1 --img 640 --cfg models/yolov5s.yaml --epochs 5 --data data/v28.yaml --weights v16.pt --evolve 30 --resume --name exp6