
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
import pdb
import json
import argparse
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import smplx
from human_body_prior.tools.model_loader import load_vposer
import ChamferDistancePytorch.dist_chamfer as ext

from cvae import HumanCVAE, ContinousRotReprDecoder
from numpy.linalg import inv
from MotionGeneration import LocalHumanDynamicsGRUNoise

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


def body_params_parse(body_params_batch):
    x_body_T = body_params_batch['transl']
    x_body_R = body_params_batch['global_orient']
    x_body_beta = body_params_batch['betas']
    x_body_pose = body_params_batch['body_pose']
    x_body_lh = body_params_batch['left_hand_pose']
    x_body_rh = body_params_batch['right_hand_pose']
    x_body_CT = body_params_batch['camera_translation']
    # print(x_body_CT)
    x_body = np.concatenate([x_body_T, x_body_R,
                             x_body_beta, x_body_pose,
                             x_body_lh, x_body_rh, x_body_CT], axis=-1)
    return x_body


def get_contact_id(body_segments_folder, contact_body_parts=['L_Hand', 'R_Hand']):

    contact_verts_ids = []
    contact_faces_ids = []

    for part in contact_body_parts:
        with open(os.path.join(body_segments_folder, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
            contact_faces_ids.append(list(set(data["faces_ind"])))

    contact_verts_ids = np.concatenate(contact_verts_ids)
    contact_faces_ids = np.concatenate(contact_faces_ids)


    return contact_verts_ids, contact_faces_ids

def convert_to_6D_rot(x_batch):
    xt = x_batch[:,:3]
    xr = x_batch[:,3:6]
    xb = x_batch[:, 6:]

    xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
    xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

    return torch.cat([xt, xr_repr, xb], dim=-1)


def convert_to_3D_rot(x_batch):
    xt = x_batch[:,:3]
    xr = x_batch[:,3:9]
    xb = x_batch[:,9:]

    xr_mat = ContinousRotReprDecoder.decode(xr) # return [:,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

    return torch.cat([xt, xr_aa, xb], dim=-1)



def verts_transform(verts_batch, cam_ext_batch):

    print(verts_batch.size())
    print(verts_batch[10,10,:])
    verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
    print(verts_batch_homo.size())
    print(verts_batch_homo[10,10,:])
    print(cam_ext_batch.size())  
    verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                               cam_ext_batch.permute(0,2,1))

    verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
    
    return verts_batch_transformed





class FittingOP:
    def __init__(self, fittingconfig, lossconfig, num_body):


        for key, val in fittingconfig.items():
            setattr(self, key, val)


        for key, val in lossconfig.items():
            setattr(self, key, val)

        self.batch_size = num_body
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')
        self.body_mesh_model = smplx.create(self.human_model_path, model_type='smplx',
                                       gender='neutral', ext='npz',
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
                                       create_transl=True,
                                       batch_size=self.batch_size
                                       )
        self.vposer.to(self.device)
        self.body_mesh_model.to(self.device)
        print(self.scene_verts_path)
        scene_o3d = o3d.io.read_triangle_mesh(self.scene_verts_path)
        scene_verts = np.asarray(scene_o3d.vertices)
        self.s_verts_batch = torch.tensor(scene_verts, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_verts_batch = self.s_verts_batch.repeat(self.batch_size, 1,1)
        # self.camera_ext = extract_ext(self.camera_path)

        self.body_rotation_rec = Variable(torch.randn(num_body,75).to(self.device), requires_grad=True)
        self.optimizer = optim.Adam([self.body_rotation_rec], lr=self.init_lr_h)


    def extract_ext(self,body_data):
        lines = [line.rstrip('\n') for line in open(self.camera_path)]
        cam_ext_list=[]
        cam_transl_batch=body_data[:,-3:].detach().cpu().numpy().squeeze()
        print(cam_transl_batch.shape)
        num=len(lines)
        for i in range(num):
            line = lines[i]
            items = line.split(' ')

            qvec = np.array([float(items[1]),float(items[2]),float(items[3]),float(items[4])])
            tvec = np.array([float(items[5]),float(items[6]),float(items[7])])
            romat = qvec2rotmat(qvec)
            cam_ext = np.eye(4)
            cam_ext[:3, 3] = tvec
            cam_ext[0:3, 0:3] =romat
            cam_ext = inv(cam_ext)
            camera_pose = np.eye(4)
            camera_transl = cam_transl_batch[i,:]
            camera_pose[:3, 3] = camera_transl
            cam_ext = np.matmul(cam_ext,(camera_pose))
            cam_ext = torch.tensor(cam_ext, dtype=torch.float32).cuda()
            cam_ext_list.append(cam_ext)

        return cam_ext_list

    def cal_loss(self, body_data_rotation):
        ### reconstruction loss
        cam_ext_list=self.extract_ext(body_data_rotation)     
        cam_ext_batch = torch.stack(cam_ext_list, dim=0)  
        print(cam_ext_batch.size())
        loss_rec = self.weight_loss_rec*F.l1_loss(body_data_rotation, self.body_rotation_rec)

        body_rec = convert_to_3D_rot(self.body_rotation_rec)
        vposer_pose = body_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)

        body_param_rec = HumanCVAE.body_params_encapsulate_batch(body_rec)
        print(body_param_rec['body_pose_vp'].size())
        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)
        print(joint_rot_batch.size())
        body_param_ = {}
        for key in body_param_rec.keys():
            if key in ['body_pose_vp']:
                continue
            else:
                body_param_[key] = body_param_rec[key]

        smplx_output = self.body_mesh_model(return_verts=True, 
                                              body_pose=joint_rot_batch,
                                              **body_param_)
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch = verts_transform(body_verts_batch, cam_ext_batch)

        vid, fid = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=self.contact_part)
        body_verts_contact_batch = body_verts_batch[:, vid, :]

        dist_chamfer_contact = ext.chamferDist()
        print(body_verts_contact_batch.contiguous())
        print(self.s_verts_batch.contiguous())
        contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch.contiguous(), 
                                                self.s_verts_batch.contiguous())
        print(contact_dist)
        loss_contact = self.weight_contact * torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0))  
        print(loss_contact)



        return loss_rec, loss_vposer

    def smoothing_loss(self,body_data_rotation):
        # print(self.body_rotation_rec.shape)
        loss_smoothing = torch.mean((self.body_rotation_rec[0:-1,:]-self.body_rotation_rec[1:,:])**2)
        # print(self.body_rotation_rec[2,9:75])
        return loss_smoothing

    def fitting(self, body_data):

        body_data_rotation = convert_to_6D_rot(body_data)

        self.body_rotation_rec.data = body_data_rotation.clone()

        body_data_rotation=body_data_rotation.detach()
        for ii in range(self.num_iter):

            self.optimizer.zero_grad()
            loss_rec, loss_vposer = self.cal_loss(body_data_rotation)
            loss_smoothing = self.smoothing_loss(body_data_rotation)
            loss = loss_rec + loss_smoothing*1.5
                        # print(self.body_rotation_rec[0,75:78])
            # if self.verbose:
            print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, total_loss={:f}'.format(
                                    ii, loss_rec.item(), loss_vposer.item(), 
                                    loss_smoothing.item(), loss.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()

        print('[INFO][fitting] fitting finish, returning optimal value')
        body_rec =  convert_to_3D_rot(self.body_rotation_rec)

        return body_rec

    def save_result(self, body_rec, fit_path):


        if not os.path.exists(fit_path):
            os.makedirs(fit_path)
            print(fit_path)
        # print(body_rec.shape)
        body_param_list = HumanCVAE.body_params_encapsulate(body_rec)
        # print('[INFO] save results to: '+output_data_file)
        # for ii, body_param in enumerate(body_param_list):

        for i in range(len(body_param_list)):
            outputfile=fit_path+'/body_gen_'+str(i).zfill(6)+'.pkl'
            outfile = open(outputfile, 'wb')
            pickle.dump(body_param_list[i], outfile)
            outfile.close()

if __name__=='__main__':


    body_path = sys.argv[1]
    fit_path = body_path.replace('body_gen','smoothed_body')

    sample_name = body_path.split('/')[-2]
    fittingconfig={
                # 'input_data_file': input_data_file,
                # 'output_data_file': os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii)),
                # 'output_data_file': '/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_baseline_postproc/realcams/'+scenename+'/body_gen_{:06d}.pkl'.format(ii),
                'scene_verts_path': '/home/miao/data/rylm/segmented_data/'+sample_name+'/meshed-poisson.ply',
                'camera_path': '/home/miao/data/rylm/segmented_data/'+sample_name+'/camerapose.txt',
                'human_model_path': './models',
                'vposer_ckpt_path': './vposer/',
                'init_lr_h': 0.003,
                'num_iter': 1,
                'batch_size': 1, 
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'contact_id_folder':  '/home/miao/data/rylm/body_segments',
                'contact_part': ['L_Leg','R_Leg'],
                'verbose': False
            }


    lossconfig={
        'weight_loss_rec': 1,
        'weight_loss_vposer':0.001,
        'weight_contact': 0.1,
        'weight_collision' : 0.5
    }

    body_gen_list = sorted(glob.glob(os.path.join(body_path, 'results/*/*.pkl')))
    
    
    # print(gen_path)
#for gen_path in gen_paths:
    # body_gen_list = sorted(glob.glob(os.path.join(gen_path, '*.pkl')))
    smplifyx_list=[]
    for body_gen in body_gen_list:
        input_data_file = body_gen
        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)
            # print(body_gen)
        # print(body_param_input)
        x =body_params_parse(body_param_input)
        # print(x.shape)
        smplifyx_list.append(x)

    smplifyx_data = np.vstack(smplifyx_list)
    # print(smplifyx_data.shape)
    body_gpu = torch.tensor(smplifyx_data, dtype=torch.float32).cuda()
    num_body = smplifyx_data.shape[0]
    fop = FittingOP(fittingconfig, lossconfig, num_body)

    body_rec = fop.fitting(body_gpu)
    out_path = fit_path
    # print(out_path)
    fop.save_result(body_rec, out_path)
        # break