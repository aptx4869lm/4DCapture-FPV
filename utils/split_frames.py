###This script is used to recode videos.
import argparse
import os
import glob
import time
import math
import shutil
import json

# for multi process
import subprocess
from multiprocessing import current_process

img_array = []
video_folders = sorted(glob.glob('./video_frames/*/'))

for video_folder in video_folders:
    frames=sorted(glob.glob(video_folder+'images/*.jpg'))
    video_length = len(frames)

    segments = int(video_length/300)
    dumped_frames = video_length-segments*300
    video_name=video_folder.split('/')[2]
    print(video_name)
    print(segments)
    start = int(dumped_frames/2)
    for i in range(segments):
        segment_name=video_name+'-'+str(i)
        new_folder = './segmented_data/'+segment_name+'/images'

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        for j in range(300):
            img = frames[start+i*300+j]

            new_img = new_folder+'/'+str(j).zfill(6)+'.jpg'
            command = 'cp'+' '+img+' '+new_img
            # print(command)
            output = subprocess.check_output(command, shell=True,
                                    stderr=subprocess.STDOUT)
