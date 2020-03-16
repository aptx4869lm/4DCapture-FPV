"""Scripts for extract frames from video"""

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

def main():
  pointcloud_file = './sample1/images.txt'
  lines = [line.rstrip('\n') for line in open(pointcloud_file)]
  lines = lines[4:]
  # print(lines[0])
  text_file = open('./sample1/camerapose.txt', 'w')

  for line in lines:
    items = line.split(' ')
    # print(line)
    if 'png' in items[-1]:
      text_file.write(' '+items[1]+' '+items[2]+' '+items[3]+' '+items[4]+' '+items[5]+' '+items[6]+' '+items[7]+'\n')
    # break
  text_file.close()
    # break
  # results_list = sorted(glob.glob(os.path.join(results_folder, '*/*.pkl')))

  # for i in range(len(results_list)):
  #   old_file = results_list[i]
  #   new_file = results_list[i].split('/')[-2]+'.pkl'
  #   new_file=new_file.replace('img','')
  #   # print(new_file)

  # #   old_file = openpose_list[i]
  # #   new_file = openpose_list[i].split('/')[-1].split('_')[-1]
    
  # #   new_file='img'+str(i).zfill(4)+'_'+new_file
  # #   # print(new_file)
  #   new_file = gen_folder+new_file


  #   command = 'mv'+' '+old_file+' '+new_file
  #   print(command)
  #   try:
  #      output = subprocess.check_output(command, shell=True,
  #                                       stderr=subprocess.STDOUT)
  #   except subprocess.CalledProcessError as err:
  #      return status, err.output
       


if __name__ == '__main__':
    description = 'Helper script for dumping video frames.'
    main()
