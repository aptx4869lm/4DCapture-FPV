
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

from MotionGeneration import LocalHumanDynamicsGRUNoise


def body_params_parse(body_params_batch):
    x_body_T = body_params_batch['transl']
    x_body_R = body_params_batch['global_orient']
    x_body_beta = body_params_batch['betas']
    x_body_pose = body_params_batch['pose_embedding']
    x_body_lh = body_params_batch['left_hand_pose']
    x_body_rh = body_params_batch['right_hand_pose']
    x_body = np.concatenate([x_body_T, x_body_R,
                             x_body_beta, x_body_pose,
                             x_body_lh, x_body_rh], axis=-1)
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
    verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
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

        self.body_rotation_rec = Variable(torch.randn(num_body,75).to(self.device), requires_grad=True)
        self.optimizer = optim.Adam([self.body_rotation_rec], lr=self.init_lr_h)

    def cal_loss(self, body_data_rotation):
        ### reconstruction loss
        loss_rec = self.weight_loss_rec*F.l1_loss(body_data_rotation, self.body_rotation_rec)
        body_rec = convert_to_3D_rot(self.body_rotation_rec)
        vposer_pose = body_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)
        return loss_rec, loss_vposer

    def smoothing_loss(self):
        loss_smoothing = F.l1_loss(self.body_rotation_rec[0:-1,:],self.body_rotation_rec[1:,:])
        return loss_smoothing



    def fitting(self, body_data):

        body_data_rotation = convert_to_6D_rot(body_data)

        self.body_rotation_rec.data = body_data_rotation.clone()


        for ii in range(self.num_iter):

            self.optimizer.zero_grad()
            loss_rec, loss_vposer = self.cal_loss(body_data_rotation)
            loss_smoothing = self.smoothing_loss()
            loss = loss_rec + loss_vposer + loss_smoothing

            # if self.verbose:
                # print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, total_loss={:f}'.format(
                #                         ii, loss_rec.item(), loss_vposer.item(), 
                #                         loss_smoothing.item(), loss.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()

        print('[INFO][fitting] fitting finish, returning optimal value')
        body_rec =  convert_to_3D_rot(self.body_rotation_rec)

        return body_rec

    def save_result(self, body_rec, fit_path):

        if not os.path.exists(fit_path):
            os.makedirs(fit_path)

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
    fit_path = sys.argv[2]

    sample_name = 'sample1'
    fittingconfig={
                # 'input_data_file': input_data_file,
                # 'output_data_file': os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii)),
                # 'output_data_file': '/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_baseline_postproc/realcams/'+scenename+'/body_gen_{:06d}.pkl'.format(ii),
                'scene_verts_path': '/is/cluster/work/yzhang/PROX/scenes_downsampled/'+sample_name+'.ply',
                'scene_sdf_path': '/is/cluster/work/yzhang/PROX/scenes_sdf/'+sample_name,
                'human_model_path': './models',
                'vposer_ckpt_path': './vposer/',
                'init_lr_h': 0.005,
                'num_iter': 500,
                'batch_size': 1, 
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'contact_id_folder': '/is/cluster/work/yzhang/PROX/body_segments',
                'contact_part': ['back','butt','L_Hand','R_Hand','L_Leg','R_Leg','thighs'],
                'verbose': False
            }


    lossconfig={
        'weight_loss_rec': 1,
        'weight_loss_vposer':0.001,
        'weight_contact': 0.1,
        'weight_collision' : 0.5
    }

    gen_paths = glob.glob(os.path.join(body_path, '*/'))
    
    

    for gen_path in gen_paths:
        body_gen_list = sorted(glob.glob(os.path.join(gen_path, '*.pkl')))
        smplifyx_list=[]
        for ii in range(len(body_gen_list)):
            input_data_file = os.path.join(gen_path,'body_gen_{:06d}.pkl'.format(ii))
            with open(input_data_file, 'rb') as f:
                body_param_input = pickle.load(f)
            x =body_params_parse(body_param_input['body'])
            smplifyx_list.append(x)

        smplifyx_data = np.vstack(smplifyx_list)

        body_gpu = torch.tensor(smplifyx_data, dtype=torch.float32).cuda()
        num_body = smplifyx_data.shape[0]
        fop = FittingOP(fittingconfig, lossconfig, num_body)

        body_rec = fop.fitting(body_gpu)
        out_path = fit_path+'/'+gen_path.split('/')[-2]
        print(out_path)
        fop.save_result(body_rec, out_path)
        # break