
import cv2
import numpy as np
import glob
 
imgs = sorted(glob.glob('./img/*.png'))
count =1
for i in range(1,len(imgs),10):
	new_img = imgs[i]
