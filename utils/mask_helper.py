"""Scripts for extract frames from video"""

import argparse
import os
import glob
import time
import math
import shutil
import json
import sys
import cv2
# for multi process
import subprocess
from multiprocessing import current_process
import numpy as np
def main(data_folder):
  
  img_folder = data_folder+'/images/'
  key_points_folder = data_folder+'/key_points/'
  # print(img_folder)
  imgs = sorted(glob.glob(img_folder+ '*.jpg'))
  # print(imgs)

  key_points = sorted(glob.glob(key_points_folder+ '*.json'))
  # print(key_points)
  
  for i in range(300):
  	key_point = key_points[i]
  	img_file = imgs[i]
  	img = cv2.imread(img_file)
  	height, width, layers = img.shape
  	with open(key_point) as keypoint_file:
  	  # print(key_point)
  	  data = json.load(keypoint_file)

  	  people = data['people'][0]
  	  body_kp = np.array(people['pose_keypoints_2d'],dtype=np.float32)
  	  body_kp = body_kp.reshape([-1, 3])
  	  
  	  body_kp = body_kp[body_kp[:,2]!=0]
  	  # print(body_kp)
  	  # print(np.min(body_kp[:,0]))
  	  # print(np.min(body_kp[:,1]))
  	  # print(np.max(body_kp[:,0]))
  	  # print(np.max(body_kp[:,1]))
  	  ul_x = int(np.min(body_kp[:,0])*0.95)
  	  ul_y = int(np.min(body_kp[:,1])*0.8)
  	  dr_x = int(np.max(body_kp[:,0])*1.05)
  	  dr_y = int(np.max(body_kp[:,1])*1.2)
  	  if dr_x>1279:
  	  	dr_x=1279
  	  if dr_y>719:
  	  	dr_y=719

  	  mask = np.ones((height,width))*255
  	  mask[ul_y:dr_y,ul_x:dr_x]=0
  	  # img = cv2.circle(img, (ul_x,ul_y),10, (0,255,0))
  	  # img = cv2.circle(img, (ul_x,dr_y),10, (0,255,0))
  	  # img = cv2.circle(img, (dr_x,ul_y),10, (0,255,0))
  	  # img = cv2.circle(img, (dr_x,dr_y),10, (0,255,0))
  	cv2.imwrite(img_file+'.png',mask)
  	# cv2.imshow('test',mask)
  	# cv2.waitKey()
  	# break
				# body_keypoints = np.array(person_data['pose_keypoints_2d'],dtype=np.float32)
				# body_keypoints = body_keypoints.reshape([-1, 3])
				# print(body_keypoints)
  # break



if __name__ == '__main__':
  data_folder=sys.argv[1]
  main(data_folder)
