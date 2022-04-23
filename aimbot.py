import math, time, cv2, datetime
import keyboard 
import mss.tools 
import numpy as np  
import torch 
import win32api, win32con, win32gui
import pyautogui as pag

# model = torch.hub.load('C:/Users/Administrator/yolov5', 'custom', path='exp20.pt', source='local', force_reload=True) 
model = torch.hub.load('C:/Users/Administrator/yolov5', 'custom', path='v28.pt', source='local', autoshape=False) 
model.conf = 0.7
model.iou = 0.6  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.amp = False  # Automatic Mixed Precision (AMP) inference
model.classes = [0,2] # 적, 테스트봇

AIMING_POINT = 0  # 0 for "head", 1 for chest, 2 for legs
# cv2.namedWindow('test',cv2.WINDOW_NORMAL)
who = { '0' : '적', '1' : '팀', '2' : '테스트봇'}
key_c =  win32api.GetKeyState(0x43)
with mss.mss() as sct:
    # Use the first monitor, change to desired monitor number 
    dimensions = sct.monitors[1] 
 
    SQUARE_SIZE = 640
    half = SQUARE_SIZE/2
    # Part of the screen to capture 
 
    monitor = {"top": int((dimensions['height'] / 2) - (SQUARE_SIZE / 2)), 
               "left": int((dimensions['width'] / 2) - (SQUARE_SIZE / 2)), 
               "width": SQUARE_SIZE, 
               "height": SQUARE_SIZE} 

    while True: 
        time_start = time.time() # 코드 실행 시간 측정 

        BRGframe = np.array(sct.grab(monitor)) 

        RGBframe = BRGframe[:, :, [2, 1, 0]] 
        # RGBframe = BRGframe

        # results = model(RGBframe, size=640)
        results = model(RGBframe)
        
        # READING OUTPUT FROM MODEL AND DETERMINING DISTANCES TO ENEMIES FROM CENTER OF THE WINDOW 

        # Get number of enemies / num of the rows of .xyxy[0] array 

        enemyNum = results.xyxy[0].shape[0] 
        enemy = results.pandas().xyxy[0]
        if enemyNum == 0: 
            pass 
        else: 
            # print(enemy)
            x1 = enemy['xmin'][0]
            x2 = enemy['xmax'][0] 
            y1 = enemy['ymin'][0] 
            y2 = enemy['ymax'][0] 

            Xenemycoord = (x2 - x1) / 2 + x1 
            Yenemycoord = (y2 - y1) / 2 + y1 
            if AIMING_POINT == 0:
                yoffset = (y2 - y1)/3
            elif AIMING_POINT == 1:
                yoffset = (y2 - y1)/5
            # MOVING THE MOUSE 
            # xoffset = (x2 - x1)/2
            difx = int(Xenemycoord - (SQUARE_SIZE / 2) ) 
            dify = int(Yenemycoord - (SQUARE_SIZE / 2) - yoffset) 
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, difx, dify, 0, 0)
            # keyboard.press('c')
            # keyboard.release('c')
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, difx, difx, 0, 0)
            # time.sleep(0.01)
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, difx, difx, 0, 0)
            # time.sleep(0.01)

            # results.display(render=True) 
            # cv2.imshow('test', results.imgs[0]) 
            # cv2.waitKey(1)
            
            t =  enemy['name'][0] # 감지 대상
            pre = int( (enemy['confidence'][0]) * 100)
            now_time = datetime.datetime.today().strftime('%m_%d_%H_%M_%S')
            print(f'{t} : {pre}% {time.time() - time_start}초')
            cv2.imwrite(f'image/{now_time}_{pre}.png',BRGframe)
            time.sleep(0.01)

        #TESTING 

        #Display the picture 

        # results.display(render=True) 

        # cv2.imshow('test', results.imgs[0]) 

        # cv2.waitKey(1)
        #Press "q" to quit 

        #if cv2.waitKey(1) & 0xFF == ord("q"): 

            #cv2.destroyAllWindows() 

            #sct.close()
