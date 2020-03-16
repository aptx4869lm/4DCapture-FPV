###This script is used to pack videos frames into videos. (For faster openpose)
import cv2
import numpy as np
import glob
 
videos_list = sorted(glob.glob('./packed_data/*/'))

for video in videos_list:
	print(video)
	video_frames = sorted(glob.glob(video+'images/*.jpg'))
	img_array = []
	for filename in video_frames:
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)	

	name = video.split('/')[-2]+'.mp4'
	name = './recoded_videos/'+name
	out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
