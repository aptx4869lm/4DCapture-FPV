
'''
this is not the original prox fitting, but a modified version just for post-processing
our generated results. The input is the smplx body parameters, and the optimization 
is based on the scene sdf and the contact loss
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import sys, os, glob
import os.path as osp
import pdb
import json
import argparse
import numpy as np
import open3d as o3d
import cv2
import PIL.Image as pil_img
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import open3d as o3d

import smplx
from human_body_prior.tools.model_loader import load_vposer
import ChamferDistancePytorch.dist_chamfer as ext
import trimesh
from cvae import HumanCVAE, ContinousRotReprDecoder
import pyrender
# from batch_gen_hdf5 import BatchGeneratorWithSceneMesh
from numpy.linalg import inv
import numpy as np

def update_cam(cam_param,  trans):
    
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat
    
    return cam_param

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def main(fitting_dir,flag):
    # fitting_dir ='/home/miao/data/rylm/packed_data/miao_corridor_0/'
    vposer_ckpt_path = './vposer/'
    body_mesh_model = smplx.create('./models', 
                                       model_type='smplx',
                                       gender='male',
                                       num_pca_comps=12,
                                       create_global_orient=True,
                                       create_body_pose=True,
                                       create_betas=True,
                                       create_left_hand_pose=True,
                                       create_right_hand_pose=True,
                                       create_expression=True,
                                       create_jaw_pose=True,
                                       create_leye_pose=True,
                                       create_reye_pose=True,
                                       create_transl=False,
                                       batch_size=1
                                       )

    vposer, _ = load_vposer(vposer_ckpt_path, vp_model='snapshot')
    ### setup visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720,visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face=True

    # scene_name = 'BasementSittingBooth'
    ## read scene mesh from file
    point_cloud=fitting_dir+'meshed-poisson.ply'
    scene = o3d.io.read_point_cloud(osp.join(point_cloud))
    vis.add_geometry(scene)
    vis.update_geometry()
    print(flag)
    
    ## dumb trans for dry run

    ## read cam trans from file
    lines = [line.rstrip('\n') for line in open(fitting_dir+'camerapose.txt')]

    items = lines[0].split(' ')

    qvec = np.array([float(items[1]),float(items[2]),float(items[3]),float(items[4])])
    tvec = np.array([float(items[5]),float(items[6]),float(items[7])])
    romat = qvec2rotmat(qvec)

    camera_lines = [line.rstrip('\n') for line in open(fitting_dir+'cameras.txt')]
    camera_lines=camera_lines[-1:]

    world_trans = np.eye(4)
    world_trans[0:3, 0:3] =romat
    world_trans[:3, 3] = tvec
    world_trans = inv(world_trans)
    # print(lines)
    # print(trans)
    body = o3d.geometry.TriangleMesh()
    vis.add_geometry(body)

    cv2.namedWindow('frame')

    if flag == 'True':
        out = 'moving_render'
        print('moving camrea')
    else:
        out = 'render'
        print('fixed camera')

    outrenderfolder = fitting_dir+out
    if not os.path.exists(outrenderfolder):
        os.makedirs(outrenderfolder)

    count = 0
    for img_name in sorted(glob.glob(os.path.join(fitting_dir, 'smoothed_body/*.pkl'))):
    # for img_name in sorted(glob.glob((fitting_dir+ 'body_gen/results/*/*.pkl'))):
        print('viz frame {}'.format(img_name))

        imgid = int(img_name.split('/')[-1].replace('.pkl','').replace('body_gen_',''))
        items = lines[count].split(' ')
        # img_name = '/home/miao/data/rylm/segmented_data/miao_mainbuilding_0-1/smoothed_body/body_gen_000299.pkl'
        qvec = np.array([float(items[1]),float(items[2]),float(items[3]),float(items[4])])
        tvec = np.array([float(items[5]),float(items[6]),float(items[7])])
        romat = qvec2rotmat(qvec)
        body_trans = np.eye(4)
        body_trans[:3, 3] = tvec
        body_trans[0:3, 0:3] =romat
        body_trans = inv(body_trans)
        # print(body_trans)
        if flag == 'True':
            # print('moving camera')
            world_trans=body_trans
        # print(body_trans)
        with open(img_name, 'rb') as f:
            param = pickle.load(f)
            # print(param)
            pose_embedding= param['body_pose']
            camera_transl = param['camera_translation']
            scale = param['scale']



        camera_pose = np.eye(4)
        camera_transl = camera_transl.squeeze()
        # camera_transl[0] *=-1.0
        camera_pose[:3, 3] = camera_transl*scale
        # print(inv(camera_pose))
        body_trans = np.matmul(body_trans,camera_pose)
        # print(camera_transl)
        pose_embedding = torch.tensor(pose_embedding, dtype=torch.float32)
        body_pose = vposer.decode(pose_embedding,output_type='aa').view(1, -1) 
        body_dict={}
        for key, val in param.items():
            if key in [
                       'jaw_pose', 'leye_pose','reye_pose',',expression','camera_translation']:
                continue
            else:
                body_dict[key]=torch.tensor(param[key], dtype=torch.float32)
        body_dict['body_pose']=body_pose
        # print(body_dict)
        model_output = body_mesh_model(return_verts=True, **body_dict)
        body_verts_np = model_output.vertices.detach().cpu().numpy().squeeze()
        body_verts_np = body_verts_np*scale
        body.vertices = o3d.utility.Vector3dVector(body_verts_np)
        body.triangles = o3d.utility.Vector3iVector(body_mesh_model.faces)
        body.vertex_normals = o3d.utility.Vector3dVector([])
        body.triangle_normals = o3d.utility.Vector3dVector([])
        body.compute_vertex_normals()
        body.transform(body_trans)
        vis.update_geometry()


        vis.run()

        count+=1

        # break




if __name__=='__main__':
    fitting_dir = sys.argv[1]
    flag = sys.argv[2]
    main(fitting_dir,flag)