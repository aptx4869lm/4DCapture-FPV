# (RYLM) Read You Like a "Mesh":Interactive Body Capture in First Person Video

## Data Preparation:
Step 0: The original video is names as "people_place_ID" (e.g.g miao_mainbuilding_0)

Step 1: Dump video frames with desired fps (30) with utils/dump_videos.py, and repack frames to videos with utils/pack_videos.py (This is for faster openpose I/O).

Step 2: Run openpose_call.py under openpose folder to get human body keypoints, then run utils/openpose_helper to rename keypoint.json and run utils/openpose_filter.py to keep the most confident human keypoints.

Step 3: Run Smplify-X model with specified focal length and data directory. 
```shell
python3 smplifyx/main.py --config cfg_files/fit_smplx.yaml  --data_folder /home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1 --output_folder /home/miao/data/rylm/downsampled_frames/miao_mainbuilding_0-1/body_gen --visualize="False" --model_folder ./models --vposer_ckpt ./vposer --part_segm_fn smplx_parts_segm.pkl --focal_length 694.0
```
TODO: Add keyboard input to helper scripts. Add example shell file fo rrunning smplify-x model.
