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
  results_folder = './results'
  gen_folder = './body_gen/'

  results_list = sorted(glob.glob(os.path.join(results_folder, '*/*.pkl')))

  print(results_list)

  for i in range(len(results_list)):
    old_file = results_list[i]
    new_file = results_list[i].split('/')[-2]+'.pkl'
    new_file=new_file.replace('img','')

    new_file = gen_folder+new_file

    command = 'mv'+' '+old_file+' '+new_file
    print(command)
    try:
       output = subprocess.check_output(command, shell=True,
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
       return status, err.output
       


if __name__ == '__main__':
    description = 'Helper script for dumping video frames.'
    main()
