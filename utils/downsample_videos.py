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
videos = sorted(glob.glob('./original_data/*.MP4'))

for video in videos:
	outputname = video.replace('original_data','recode_videos')

	command = 'ffmpeg'+' '+'-i'+' '+video+' '+'-filter:v'+' '+'fps=fps=5'+' '+outputname
	print(command)
	output = subprocess.check_output(command, shell=True,
	                                stderr=subprocess.STDOUT)

