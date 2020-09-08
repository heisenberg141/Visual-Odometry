import numpy as np 
import cv2
import os

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, "dataset/poses/01.txt")

traj = np.zeros((600,600,3), dtype=np.uint8)
images_path='data_odometry_gray/dataset/sequences/00/image_0'
files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
for img_id in range(len(files)):
         
        if img_id!=173:
            img = cv2.imread(files[img_id], 0)    
        vo.update(img, img_id)
        

#        cv2.circle(img, (st1,st2), 1, (0,0,255), 1)
        cur_t = vo.cur_t
        if(img_id > 0):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        #print('xframe_id',img_id,':',x)
        #print('yframe_id',img_id,':',z)
        #print('trueX frame_id',img_id,':',vo.trueX)
        #print('trueY frame_id',img_id,':',vo.trueZ)
        draw_x, draw_y = int(x)+290, int(z)+90
        true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90
         
        cv2.circle(traj, (draw_x,draw_y), 1, (0,0,255), 1)
        cv2.circle(traj, (true_x,true_y), 1, (255,0,0), 1)
##        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
##        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
##        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
##        for x,y in vo.px_ref:
##                cv2.circle(img, (x,y), 2, (0,0, 255), -1)
       
        cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        ch=cv2.waitKey(1)
        if ch==27:
                break
if ch==27:
        cv2.destroyAllWindows()
print('done')
cv2.imwrite('map.png', traj)
