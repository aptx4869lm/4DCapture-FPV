## Scripts for extract frames from video 
## A simplified version form Yin's script

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

from joblib import delayed
from joblib import Parallel

def extract_frames(video_file, output_folder, 
                   width, height, fps, shortest):
    """call ffmpeg to get all frames"""
    status = False

    if shortest > 0:
        # prob the video resolution
        # and keep the shortest side to shortest
        prob_cmd = ['ffprobe',
                    '-v error',
                    '-of flat=s=_',
                    '-select_streams v:0', 
                    '-show_entries stream=height,width',
                    '{:s}'.format(video_file)]
        prob_cmd = ' '.join(prob_cmd)
        width = -1
        height = -1
        try:
            output = subprocess.check_output(prob_cmd, shell=True,
                                             stderr=subprocess.STDOUT)
            lines = output.split('\n')
            if len(lines) == 3:
                str_width = lines[0].replace('streams_stream_0_width=', '')
                str_height = lines[1].replace('streams_stream_0_height=', '')
                width = int(str_width)
                height = int(str_height)

            if (width < 0) or (height < 0):
                return status, "Could not prob video resolution!"

        except subprocess.CalledProcessError as err:
            return status, err.output

        # compute the new width / height (round to factor of 2)
        ratio = shortest / float(min(width, height))
        width = int(round(ratio * width))
        width = width + width % 2
        height = int(round(ratio * height))
        height = height + height % 2

    command = ['ffmpeg',
               '-i', '{:s}'.format(video_file),
               '-r', '{:s}'.format(str(fps)),
               '-f', 'image2', '-q:v', '1',
               '-s', '{:d}x{:d}'.format(width, height),
               '{:s}/%06d.jpg'.format(output_folder)
              ]
    command = ' '.join(command)
    print(command)
    try:
       output = subprocess.check_output(command, shell=True,
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
       return status, err.output
    
    # for debug only!
    # pid = current_process()
    # print "PID: {:s} -- {:s}".format(pid, command)

    status = True
    return status, 'Finished'

def extract_frames_wrapper(video_file, ext, root_output_dir, 
                           width, height, fps, shortest):
    """wrapper for extract frames"""
    video_name = os.path.basename(video_file).split('.'+ ext)[0]
    output_folder = os.path.join(root_output_dir, video_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # each video folder holds img / flow / etc
    output_img_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(output_img_folder):
        os.mkdir(output_img_folder)
    status, log = extract_frames(video_file, output_img_folder, 
                                 width, height, fps, shortest)
    return tuple([video_file, status, log])

def main(video_dir, output_dir, 
         ext='avi', num_jobs=1, 
         width=320, height=256, fps=24,
         shortest=-1):
    """
    The main function for cropping videos in a folder
    """
    # get all video file list

    file_list = sorted(glob.glob(os.path.join(video_dir, '*.{:s}'.format(ext))))
    print(file_list)
    # for debug
    # file_list = file_list[0:100]

    # create folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # lauch jobs
    start_time = time.time()
    if num_jobs == 1:
        status_lst = []
        for video_file in file_list:
            status_lst.append(extract_frames_wrapper(
                video_file, ext, output_dir, width, height, fps, shortest))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(
            delayed(extract_frames_wrapper)(video_file, ext, output_dir, 
                                            width, height, fps, shortest) 
            for video_file in file_list)
    end_time = time.time()



if __name__ == '__main__':
    description = 'Helper script for dumping video frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('video_dir', type=str,
                   help='Root video folder with all avi videos.')
    p.add_argument('output_dir', type=str,
                   help='Output directory where cropped videos will be saved.')
    p.add_argument('-e', '--ext', type=str, default='MP4',
                   help='Video extension')
    p.add_argument('-n', '--num-jobs', type=int, default=1)
    p.add_argument('-nw', '--width', type=int, default=1280,
                   help='New image width')
    p.add_argument('-nh', '--height', type=int, default=720,
                   help='New image height')
    p.add_argument('-nf', '--fps', type=int, default=30,
                   help='Video frame rate (24 for GTEA | 25 for UCF)')
    p.add_argument('-nshort', '--shortest', type=int, default=-1,
                   help='Keep the shortest side (set to -1 for fixed sized frames)')
    main(**vars(p.parse_args()))