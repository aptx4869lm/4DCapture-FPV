import os

import cv2
import numpy as np
import glob
cmd1 = './build/examples/openpose/openpose.bin --video'
cmd2 = '--face --hand --write_json' 
cmd3 = '--write_video'



videos_list = sorted(glob.glob('/home/miao/data/rylm/recoded_videos/*.mp4'))

for video in video_list:
	print(video)

# os.system(cmd)

