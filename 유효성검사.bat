@echo on
cd C:\Users\Administrator\yolov5

python val.py --weights v26.pt --data data/v26.yaml --img 640 --batch 32
