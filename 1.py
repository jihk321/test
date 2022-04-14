import math, time, cv2
import keyboard 
import mss.tools 
import numpy as np  
import torch 
import win32api, win32con, win32gui

# model = torch.hub.load('C:/Users/Administrator/yolov5', 'custom', path='exp20.pt', source='local', force_reload=True) 
model = torch.hub.load('C:/Users/Administrator/yolov5', 'custom', path='apex_v15.pt', source='local', force_reload=True) 

AIMING_POINT = 0  # 0 for "head", 1 for chest, 2 for legs
cv2.namedWindow('test',cv2.WINDOW_NORMAL)

with mss.mss() as sct: 
 
    # Use the first monitor, change to desired monitor number 
    dimensions = sct.monitors[1] 
 
    SQUARE_SIZE = 600 

    # Part of the screen to capture 
 
    monitor = {"top": int((dimensions['height'] / 2) - (SQUARE_SIZE / 2)), 
 
               "left": int((dimensions['width'] / 2) - (SQUARE_SIZE / 2)), 
 
               "width": SQUARE_SIZE, 
 
               "height": SQUARE_SIZE} 

    while True: 
 
        # READING OUTPUT FROM THE MODEL 
        # SCREEN CAPTURE AND CONVERTING TO MODEL'S SUPPORTED FORMAT 
        # Screenshot
 
        BRGframe = np.array(sct.grab(monitor)) 
 
        # Convert to format model can read 
 
        RGBframe = BRGframe[:, :, [2, 1, 0]] 
 
  
 
        # PASSING CONVERTED SCREENSHOT INTO MODEL 
 
        results = model(RGBframe, size=600)
        model.conf = 0.7
        # model.classes = [0,1,2,4] #exp20
        model.classes = [0,2] # exp32
        # READING OUTPUT FROM MODEL AND DETERMINING DISTANCES TO ENEMIES FROM CENTER OF THE WINDOW 
 
        # Get number of enemies / num of the rows of .xyxy[0] array 
 
        enemyNum = results.xyxy[0].shape[0] 

        if enemyNum == 0: 
 
            pass 
        
        else: 
            
            # Reset distances array to prevent duplicating items 
            # print(results.pandas().xyxy[0])
            distances = [] 
            target = [] # 목표물 3 : 팀
            closest = 1000 

            # Cycle through results (enemies) and get the closest 
 
            for i in range(enemyNum): 
                # if float(results.xyxy[0][i, 5])  == 3.0 :
                #     pass
                # else :
                x1 = float(results.xyxy[0][i, 0]) 
                x2 = float(results.xyxy[0][i, 2]) 
                y1 = float(results.xyxy[0][i, 1]) 
                y2 = float(results.xyxy[0][i, 3]) 
                t =  float(results.xyxy[0][i, 5]) # 감지 대상
                pre = int((results.xyxy[0][i, 4]) * 100)
                centerX = (x2 - x1) / 2 + x1 
                centerY = (y2 - y1) / 2 + y1 

                distance = math.sqrt(((centerX - 300) ** 2) + ((centerY - 300) ** 2)) 
                print(f'{t} : {pre}%')
                distances.append(distance) 
                target.append(t)
                # Get the shortest distance from the array (distances) 
                if distances[i] < closest: 
                
                    closest = distances[i] 
                    closestEnemy = i 
                img = results.imgs[0]
                (x,y) = x1,y1
                (xx,yy) = x2,y2
                cv2.line(results.imgs[0], (int(centerX), int(centerY)), (300, 300), (255, 0, 0), 1, cv2.LINE_AA) 
                # cv2.rectangle(img,(x,y),(xx,yy),(255,255,0),2)
                # cv2.imwrite(f'image/{t}_{pre}per_{centerX}_{centerX}.jpg',img)
            # Getting the coordinates of the closest enemy 
            try : 
                x1 = float(results.xyxy[0][closestEnemy, 0]) 
                x2 = float(results.xyxy[0][closestEnemy, 2]) 
                y1 = float(results.xyxy[0][closestEnemy, 1]) 
                y2 = float(results.xyxy[0][closestEnemy, 3]) 
                if AIMING_POINT == 0:
                    y2 = (y2 / 20) * 1
        
                elif AIMING_POINT == 1:
                    y2 = (y2 / 20) * 2
        
                elif AIMING_POINT == 2:
                    y2 = (y2 / 20) * 5
                Xenemycoord = (x2 - x1) / 2 + x1 
                Yenemycoord = (y2 - y1) / 2 + y1 

                x = (x2 - x1) / 2 + x1
                y = y1 + y2

                xx = int(x - (SQUARE_SIZE / 2))
                yy = int(y - (SQUARE_SIZE / 2))
    
                # MOVING THE MOUSE 
    
                difx = int(Xenemycoord - (SQUARE_SIZE / 2)) 
                dify = int(Yenemycoord - (SQUARE_SIZE / 2)) 
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, xx, yy, 0, 0)
                time.sleep(0.05)
                # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, xx, yy, 0, 0)
                # time.sleep(0.1)
                # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, xx, yy, 0, 0)
                # time.sleep(0.1)

                results.display(render=True) 
                cv2.imshow('test', results.imgs[0]) 
                cv2.waitKey(1)
            except : pass
 
        # DISPLAYING DATA 
 
        # will make in the future 
 
  
 
        #TESTING 
 
        #Display the picture 
 
        # results.display(render=True) 
 
        # cv2.imshow('test', results.imgs[0]) 

        # cv2.waitKey(1)
        #Press "q" to quit 
 
        #if cv2.waitKey(1) & 0xFF == ord("q"): 
 
            #cv2.destroyAllWindows() 
 
            #sct.close()
