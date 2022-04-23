@echo on
cd C:\Users\Administrator\yolov5

python train.py --img 640 --batch -1 --epochs 10000 --data data/v28.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt --name v28 --hyp data/hyps/custom.yaml