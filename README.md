# (RYLM) Read You Like a "Mesh":Interactive Body Capture in First Person Video

## Data Preparation:
Step 0: The original video is names as "people_place_ID" (e.g.g miao_mainbuilding_0) 

Step 1: Dump video frames with desired fps (30) with utils/dump_videos.py. Run utils/split_frames to segment videos into equally long subatom clips.  Repack frames to videos with utils/pack_videos.py (This is for faster openpose I/O).

Step 2: Run openpose_call.py under openpose folder to get human body keypoints, then run utils/openpose_helper to rename keypoint.json and run utils/openpose_filter.py to keep the most confident human keypoints.

Step 3: Run Smplify-X model with specified focal length and data directory. This step may take up to several hours. For instance:
```shell
python3 smplifyx/main.py --config cfg_files/fit_smplx.yaml  --data_folder /home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1 --output_folder /home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1/body_gen --visualize="False" --model_folder ./models --vposer_ckpt ./vposer --part_segm_fn smplx_parts_segm.pkl --focal_length 694.0
```

Step 4: Run Colmap for to generate scene mesh and camera trajectory. This step make take up to several hours depneding on the complexity of the scene. Then Run utils/camerpose_helper and utils/pointscloud_helper.py to generate desired points cloud file and camera pose.

TODO: Add keyboard input to helper scripts. Add example shell file for running smplify-x model.

## Temporal Smoothing:
Runn global_optimization.py to conduct temproal smoothing on smplify-x outputs:
```shell
python3 global_optimization.py '/home/miao/data/rylm/packed_data/miao_mainbuidling_0-1/body_gen' '/home/miao/data/rylm/packed_data/miao_mainbuidling_0-1/smoothed_body
```

The resulting data should be organized as following:
- datafolder:
  - videoname:
    - images: folder that contains all video frames
    - keypoints: folder that contains all body keypoints
    - body_gen: folder that contains all body mesh files:
    - smoothed_boyd: folder that contains all temporal-smoothed body mesh files:
    - camera_pose.txt: text file that contains camera pose at each temporal footprint
    - meshed-poisson.ply: scene mesh file from dense reconstruction
    - camera.txt: text file that contains camera parameters
    - xyz.ply point cloud file. (use meash lab to convert .xyz file to .ply file)

## Global Transformation:
Run global_vis.py to transform the body mesh in pivot coordinate to world coordinate. By default the viewpoint of open3d is the initial position camera trajectory. Setting bool flag to 'True' will resulting into a open3d viewpoint moving the same way as camera viewer.
```shell
python3 global_vis.py '/home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1/' False
```

TODO: an interactive open3d viewer for debugging. Using calibrated camera parameters to re-run everything.
## Pack Video Outputs:
Run pack_videosoutputs.py to generate video outputs.
```shell
python3 pack_videosoutputs.py '/home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1/' 'render'
```
