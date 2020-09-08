import numpy as np 
import cv2
import os

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 750
flag=0
lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
images_path='data_odometry_gray/dataset/sequences/00/image_0'
files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.px_cal=None
        self.px_new=None
        self.frame_no=None
        self.RNT_start=None
        self.k=0
        self.tri=None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        #self.detector = cv2.xfeatures2d.SIFT_create()
        #self.detector = cv2.ORB_create()
        self.detector1 = cv2.xfeatures2d.SURF_create()
        with open(annotations) as f:
            self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

    
    def processFirstFrame(self):
        
        p1 = self.detector.detect(self.new_frame)
        p2 = self.detector1.detect(self.new_frame)
        self.px_ref=np.concatenate((p1,p2),axis=0)
        #self.px_ref=p1
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        #print(self.px_ref)
        self.px_ref=self.px_ref[self.px_ref[:,0].argsort()]
        #print(self.px_ref)
        q=self.px_ref.shape
        #print(q)
        div=q[0]/2
        div=int(div)
        for h in range(0,div,1):
            #print(h)
            if((self.px_ref[h+1][0]-self.px_ref[h][0])<=10.0):
                self.px_ref=np.delete(self.px_ref,h,axis=0)
                
        # print(self.px_ref.shape)    
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.px_ref = self.px_cur
        self.RNT_start=np.concatenate((self.cur_R,self.cur_t) , axis=1)
        #self.RNT_start=np.vstack([self.RNT_start,[0,0,0,1]])
        self.px_cal=self.px_cur
        
    def processFrame(self, frame_id):
        cam_mat=np.zeros(9)
        cam_mat=np.resize(cam_mat,(3,3))
        cam_mat[0]=[718.856, 0.0, 607.1928]
        cam_mat[1]=[0.0 ,718.856 ,185.2157]
        cam_mat[2]=[0.0, 0.0, 1.0]        
             
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        self.px_new=self.px_cur
        
        
            
        if(frame_id%5==0) and (frame_id>0):
            j=self.px_cal.shape[0]
            i=self.px_new.shape[0]
            
            

        
            if j>i:
                m=None
                m=np.arange(i,j)
                
        
                for g in range(i+1,j):
                    self.px_cal= np.delete(self.px_cal, m, axis=0)
                
##                print(self.px_new.shape)
##                print(self.px_cal)
                E2, mask2 = cv2.findEssentialMat(self.px_new,self.px_cal , focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                _, R2, t2, mask2 = cv2.recoverPose(E, self.px_new,self.px_cal , focal=self.focal, pp = self.pp)
                RNT=np.concatenate((R2,t2) , axis=1)
##                RNT=np.vstack([RNT,[0,0,0,1]])
                thresh_rat= np.subtract(self.cur_t, t2)
                self.k=thresh_rat[0]+thresh_rat[1]+thresh_rat[2]
                #print('2-',frame_id,':',k[0])

            elif j<i:
##                print('j')
                l=None
                l=np.arange(j,i)
##                print(type(self.px_new))
##                print(type(self.px_cal))
##                print(self.px_new.shape)
##                print(self.px_cal.shape)
                for g in range(j+1,i):
                    self.px_new=np.delete(self.px_new,l,axis=0)
##                print(self.px_new.shape)
                E2, mask2 = cv2.findEssentialMat(self.px_new,self.px_cal , focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                _, R2, t2, mask2 = cv2.recoverPose(E, self.px_new, self.px_cal , focal=self.focal, pp = self.pp)
                RNT=np.concatenate((R2,t2) , axis=1)
                #RNT=np.vstack([RNT,[0,0,0,1]])
                #thresh_rat=np.multiply(self.RNT_start,RNT)
                thresh_rat= np.subtract(self.cur_t, t2)
                self.k=thresh_rat[0]+thresh_rat[1]+thresh_rat[2]
                #print('2-',frame_id,':',k[0])

          
            
            if self.k>8.0:
                RNT0=np.dot(cam_mat,self.RNT_start)
                RNT1=np.dot(cam_mat,RNT)
                px0=np.resize(self.px_cal,(self.px_cal.shape[0],1,2))
                px1=np.resize(self.px_new,(self.px_new.shape[0],1,2))
                #print(self.px_new.shape)
                #print(px1.shape)
                self.tri=cv2.triangulatePoints(RNT0,RNT1,px0,px1)
                self.px_cal=self.px_new
                if frame_id<len(files):
                    kitti=cv2.imread(files[frame_id+1],0)
                p1 = self.detector.detect(kitti)
                p2 = self.detector1.detect(kitti)
                px=np.concatenate((p1,p2),axis=0)
                self.tri[0]=self.tri[0]/self.tri[3]
                self.tri[1]=self.tri[1]/self.tri[3]
                self.tri[2]=self.tri[2]/self.tri[3]
                self.tri=np.delete(self.tri,3,axis=0)
                px= np.array([x.pt for x in px], dtype=np.float32)
                po=self.tri.shape[1]
                op=px.shape[0]
                # print('tri1',self.tri.shape)
                # print('keypoints1',px.shape)
                if(po>op):
                    io=None
                    io=np.arange(op,po)
                    for g in range(op+1,po):
                        
                        self.tri.T=np.delete(self.tri.T, io, axis=0)
                elif(po<op):

                    io=None
                    io=np.arange(po,op)
                    for g in range(po+1,op):
                        
                        px=np.delete(px, io, axis=0)
                # print('tri',self.tri.shape)
                # print('keypoints',px.shape)
                ret,rvecs,tvecs=cv2.solvePnP(self.tri.T,px,cam_mat,0.0)
                # print('abhi to sahi chal raha hai code')
##            #print(np.linalg.det(thresh_rat))
        
            
        if (absolute_scale > 0.1):
                  
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

##        if(self.px_ref.shape[0] < kMinNumFeature):
##            p1 = self.detector.detect(self.new_frame)
##            p2 = self.detector1.detect(self.new_frame)
##            self.px_cur=np.concatenate((p1,p2),axis=0)
##            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

        if(self.frame_no%5==0):
            p1 = self.detector.detect(self.new_frame)
            p2 = self.detector1.detect(self.new_frame)
            self.px_cur=np.concatenate((p1,p2),axis=0)
            #self.px_cur=p1
            
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            #print('unsorted',self.px_ref)
            self.px_ref=self.px_ref[self.px_ref[:,0].argsort()]
            #print('sorted',self.px_ref)
            q=self.px_ref.shape
            #print(q)
            div=q[0]/2
            div=int(div)
            for h in range(0,div,1):
                if((self.px_ref[h+1][0]-self.px_ref[h][0])<=0.5):
                    self.px_ref=np.delete(self.px_ref,h+1,axis=0)
                
            # print(self.px_ref.shape)    
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        self.frame_no=frame_id   
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
            
        self.last_frame = self.new_frame
