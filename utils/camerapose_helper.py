
import sys
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

def main(camera_folder):
  camera_file =camera_folder+'images.txt'
  lines = [line.rstrip('\n') for line in open(camera_file)]
  lines = lines[3:]
  print(lines[0])
  output_file = camera_file.replace('images','camerapose')
  text_file = open(output_file, 'w')

  for line in lines:
    items = line.split(' ')
    # print(line)
    if 'jpg' in items[-1]:
      text_file.write(' '+items[1]+' '+items[2]+' '+items[3]+' '+items[4]+' '+items[5]+' '+items[6]+' '+items[7]+'\n')
    # break
  text_file.close()



if __name__ == '__main__':
    camera_folder=sys.argv[1]
    main(camera_folder)
