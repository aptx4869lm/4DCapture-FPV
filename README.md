# rylm
# Read You Like a "Mesh":Interactive Body Capture in First Person Video

# Data Preparation:
Step 0: The original video is names as "people_place_ID" (e.g.g miao_mainbuilding_0)

Step 1: Dump video frames with desired fps (30) with utils/dump_videos.py, and repack frames to videos with utils/pack_videos.py (This is for faster openpose I/O).

Step 2: Recode dumped video frames 
Step 1: Run Smplify-X model with specified focal length and data directory
Result should be 
