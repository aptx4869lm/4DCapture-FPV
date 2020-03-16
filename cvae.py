from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import open3d as o3d
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

import random
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import sys, os
sys.path.append(os.path.join(os.getcwd(), '/torch_mesh_isect/build/lib.linux-x86_64-3.6'))

# import dsntnn

# from mesh_intersection.bvh_search_tree import BVH
# import mesh_intersection.loss as collisions_loss
# from mesh_intersection.filter_faces import FilterFaces

# from net_layers import BodyGlobalPoseVAE, BodyLocalPoseVAE, ResBlock








################################################################################
## Conditional VAE of the human body,
## Input: 72/75-dim, [T (3d vector), R (3d/6d), shape (10d), pose (32d),
#         lefthand (12d), righthand (12d)]
## Note that, it requires pre-trained VPoser and latent variable of scene
################################################################################


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
       
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot





class HumanCVAE(nn.Module):
    '''
    NOTE: in our current implementation, we only use the static methods of
          this class
    '''

    def __init__(self,
                 latentD=64,
                 n_dim_body=72,
                 n_dim_scene=128,
                 dropout_ratio = 0
                 ):
        super(HumanCVAE, self).__init__()

        self.latentD = latentD

        n_dim_input = n_dim_body+n_dim_scene
        self.enc_fc1 = nn.Linear(n_dim_input, latentD)
        self.enc_fc2 = nn.Linear(latentD, latentD)
        self.enc_fc3 = nn.Linear(latentD, latentD)

        self.enc_mu = nn.Linear(latentD, latentD)
        self.enc_logsigma2 = nn.Linear(latentD, latentD)

        self.dec_fc1 = nn.Linear(latentD+n_dim_scene, latentD)
        self.dec_fc2 = nn.Linear(latentD, latentD)
        self.dec_fc3 = nn.Linear(latentD, n_dim_body)           


        self.dropout_ratio = dropout_ratio
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio,
                                      inplace=False)


    def swish(self, x):
        return x*F.sigmoid(x)


    def encode(self, x_body, z_s):


        Xout = torch.cat([x_body, z_s], dim=1)
       
        Xout = self.swish(self.enc_fc1(Xout))
        Xout = self.swish(self.enc_fc2(Xout))+Xout
        Xout = self.swish(self.enc_fc3(Xout))+Xout

        mu = self.enc_mu(Xout)
        logsigma2 = self.enc_logsigma2(Xout)
        return mu, logsigma2



    def decode(self, z, z_s):
        '''
        input: z: latent variable of body representation
               z_s: latent variable of scene representation
        '''

        z = torch.cat([z, z_s], dim=1)

        Xout = self.swish(self.dec_fc1(z))
        if self.dropout_ratio>0:
            Xout = self.dropout(Xout)

        Xout = self.swish(self.dec_fc2(Xout))+Xout
        if self.dropout_ratio>0:
            Xout = self.dropout(Xout)

        Xout = self.dec_fc3(Xout)
 
        return Xout


    def forward(self, x_body, eps, z_s):
        '''
        input: x_body: body representation, [batch, 72D]
               z_s: scene representation, [batch, 128D]

        '''
        mu, logsigma2 = self.encode(x_body, z_s)

        q_z_sample = eps*torch.exp(logsigma2/2.0) + mu
        x_body_rec = self.decode(q_z_sample, z_s)


        return x_body_rec, mu, logsigma2

    @staticmethod
    def body_params_encapsulate(x_body_rec):
        x_body_rec_np = x_body_rec.detach().cpu().numpy()
        n_batch = x_body_rec_np.shape[0]
        rec_list = []

        for b in range(n_batch):
            body_params_batch_rec={}
            body_params_batch_rec['transl'] = x_body_rec_np[b:b+1,:3]
            body_params_batch_rec['global_orient'] = x_body_rec_np[b:b+1,3:6]
            body_params_batch_rec['betas'] = x_body_rec_np[b:b+1,6:16]
            body_params_batch_rec['body_pose'] = x_body_rec_np[b:b+1,16:48]
            body_params_batch_rec['left_hand_pose'] = x_body_rec_np[b:b+1,48:60]
            body_params_batch_rec['right_hand_pose'] = x_body_rec_np[b:b+1,60:72]
            body_params_batch_rec['camera_translation'] = x_body_rec_np[b:b+1,72:]

            rec_list.append(body_params_batch_rec)
        # print(rec_list[-1]['transl'])
        # print(rec_list[-1]['global_orient'])
        # print(rec_list[-1]['camera_translation'])
        return rec_list



    @staticmethod
    def body_params_parse(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']

        x_body = np.concatenate([x_body_T, x_body_R,
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu


    @staticmethod
    def body_params_parse_fitting(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        # print(body_params_batch)
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']
        x_body_CT = body_params_batch['camera_translation']
        # cam_ext = torch.tensor(body_params_batch['cam_ext'], dtype=torch.float32).cuda()
        # cam_int = torch.tensor(body_params_batch['cam_int'], dtype=torch.float32).cuda()
       
        x_body = np.concatenate([x_body_T, x_body_R,
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh,x_body_CT], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu






