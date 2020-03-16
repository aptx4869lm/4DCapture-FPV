###This script is used to filter oprnpose result. 
###By default, the first people is the most confifent one
import cv2
import numpy as np
import glob
import json
videos_list = sorted(glob.glob('./packed_data/*/'))
for video in videos_list:
	keypoints_list = sorted(glob.glob(video+'key_points/*.json'))
	for keypoints in keypoints_list:
		with open(keypoints) as keypoint_file:
			data = json.load(keypoint_file)
		if(len(data['people'])>1):
			print(keypoints)
			data['people'] = [data['people'][0]]
			with open(keypoints, 'w') as outfile:
			    json.dump(data, outfile)

			# for i in range(len(data['people'])):
			# 	print(data['people'][i])
	   #  for idx, person_data in enumerate(data['people']):
				# body_keypoints = np.array(person_data['pose_keypoints_2d'],dtype=np.float32)
				# body_keypoints = body_keypoints.reshape([-1, 3])
				# print(body_keypoints)

    # break
	# break
