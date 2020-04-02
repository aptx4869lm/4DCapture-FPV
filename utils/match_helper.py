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
  imgs = sorted(glob.glob(img_folder+ '*.jpg'))
  print(len(imgs))
  # 0->60 61 70 71 80 81 90 91
  # 1->61 62 71 72 81 82 91 92
  # 2->62 63 72 73 82 83 92 93
  ###
  ###
  ###
  # 239->299
  file_name = data_folder+'/matches.txt'
  text_file = open(file_name, 'w')

  for i in range(240):

    matches = []
    matches.append(i+60)
    matches.append(i+61)
    matches.append(i+70)
    matches.append(i+71)
    matches.append(i+80)
    matches.append(i+81)
    matches.append(i+90)
    matches.append(i+91)
    img = imgs[i]
    base_name = os.path.basename(img)
    print(base_name)
    # print(matches)
    # 
    for match in matches:
      if match <= 299:
        name = str(match).zfill(4)+'.jpg'
        text_file.write(base_name+' '+name+'\n')
  text_file.close()


if __name__ == '__main__':
  data_folder=sys.argv[1]
  main(data_folder)
