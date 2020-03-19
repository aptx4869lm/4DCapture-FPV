###This script is used to pack videos frames into videos. (For faster openpose)
import cv2
import numpy as np
import glob
import sys
# video_folder = '/home/miao/data/rylm/segmented_data/miao_downtown_1-0/'

def main(video_folder,subfolder):
	video_frames = sorted(glob.glob(video_folder+subfolder+'/*.png'))
	img_array = []
	for filename in video_frames:
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)	

	name = subfolder+'.mp4'
	name =video_folder+name
	out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

if __name__=='__main__':
    video_folder=sys.argv[1]
    subfolder = sys.argv[2]
    main(video_folder,subfolder)