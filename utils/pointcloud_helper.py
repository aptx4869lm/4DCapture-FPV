"""Scripts for extract frames from video"""

import argparse
import os
import glob
import time
import math
import shutil
import json
import sys
# for multi process
import subprocess
from multiprocessing import current_process

def main(pointcloud_folder):
  pointcloud_file = pointcloud_folder+'/points3D.txt'
  lines = [line.rstrip('\n') for line in open(pointcloud_file)]
  lines = lines[3:]
  # print(lines[0])
  output_file = pointcloud_file.replace('points3D','xyz').replace('txt','xyz')
  text_file = open(output_file, 'w')
                
  for line in lines:
    items = line.split(' ')

    text_file.write(' '+items[1]+' '+items[2]+' '+items[3]+' '+items[4]+' '+items[5]+' '+items[6]+'\n')
  text_file.close()


if __name__ == '__main__':
    pointcloud_folder=sys.argv[1]
    main(pointcloud_folder)
