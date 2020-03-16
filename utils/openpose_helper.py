#### this script is used to rename the openpose output for smplify-x usage
import cv2
import numpy as np
import glob
import json
import subprocess
from multiprocessing import current_process
videos_list = sorted(glob.glob('./packed_data/*/'))
for video in videos_list:
	keypoints_list = sorted(glob.glob(video+'key_points/*.json'))
	for keypoints in keypoints_list:
		items = keypoints.split('_')
		name = int(items[-2])+1
		name=str(name).zfill(6)+'_keypoints.json'
		items = keypoints.split('/')
		new_name=('/').join(items[:-1])+'/'+name
		command='mv'+' '+keypoints+' '+new_name
		print(command)
		output = subprocess.check_output(command, shell=True,
	                                        stderr=subprocess.STDOUT)