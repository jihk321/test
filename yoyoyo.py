# import base64
# from traceback import print_tb
from typing import Type
import torch
import numpy as np
import pyautogui
import win32api, win32con, win32gui
import cv2
import math
import time
from PIL import ImageGrab
import pandas as pd
from PIL import ImageGrab


def aiming(result):
    x1 = result['xmin']
    y1 = result['ymin']
    x2 = result['xmax']
    y2 = result['ymax']
    
    c_x = ((((x2 - x1) / 2 ) + x1 ) - img_w/2 ) * 2 
    c_y = (((y2 - y1)/4  - (img_h/2) + y1) ) * 1.5

    x = int(c_x)
    y = int(c_y)

    if abs(x) < 2 and abs(y) < 2 :
        pass
    else :
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)

    # print(f'{x}만큼 이동했음 좌{x1} 우{x2} 상{y1} 하{y2} ' )
    

    predi = int(result['confidence'] * 100)
    name = result['name'] 
    print(f'{predi}% {name} 감지 / {x} {y} 에임 조정 {time.time() - time_start}초')
    # win32api.keybd_event(win32con.VK_F11,0)
    
    time.sleep(0.01)

detector = torch.hub.load('','custom',path= 'exp20.pt',source='local', force_reload=True)
cv2.namedWindow('test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('test',640,360)

while True:
    time_start = time.time() # 코드 실행 시간 측정
    # Get rect of Window
    hwnd = win32gui.FindWindow(None, 'Apex Legends')
    #hwnd = win32gui.FindWindow("UnrealWindow", None) # Fortnite
    rect = win32gui.GetWindowRect(hwnd)
    region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

    # Get image of screen
    # ori_img = np.array(pyautogui.screenshot(region=region))
    image = np.array(pyautogui.screenshot(region=region))
    # image = np.array(ImageGrab.grab())
    img = cv2.resize(image,(640,640))
    # img = image
    img_w, img_h = img.shape[1],img.shape[0]

    # Detection
    result = detector(img)
    enemynum = result.xyxy[0].shape[0]
    result = result.pandas().xyxy[0]

    if enemynum != 0 :
        # for i in range(len(result.index)) :
        if result['name'][0] == 'test' :
            if result['confidence'][0] > 0.5 :
                aiming(result.iloc[0])
            
        elif result['name'][0] == 'enemy' :
            xmin = result['xmin'][0]
            xmax = result['xmax'][0]
            ymin = result['ymin'][0]
            print(f'{xmax-xmin} 크기')
            if result['confidence'][0] < 0.7 :
                if 760 < xmin < 1160 | 440 < ymin < 640 :
                    aiming(result.iloc[0])
            elif 0.7 < result['confidence'][0] < 0.8 :
                if 560 < xmin < 1360 | 240 < ymin < 840:
                    aiming(result.iloc[0])
            elif result['confidence'][0] > 0.8 : 
                aiming(result.iloc[0])

        elif result['name'][0] == 'skil' :
            if result['confidence'][0] > 0.7 :
                aiming(result.loc[0])
        elif result['name'][0] == 'detect' :
            if result['confidence'][0] > 0.6 :
                aiming(result.loc[0])
 
    # print("time :", time.time() - time_start)
        # for i in range(len(result.index)) :
        (x,y) = int(result['xmin'][0]),int(result['ymin'][0])
        (xx,yy) = int(result['xmax'][0]),int(result['ymax'][0])
        cv2.rectangle(img,(x,y),(xx,yy),(255,255,0),2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('test',img)
        cv2.waitKey(1)
        # 표시