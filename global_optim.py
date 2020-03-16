
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
    def __init__(self, fittingconfig, lossconfig):


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

        self.xhr_rec = Variable(torch.randn(self.batch_size,75).to(self.device), requires_grad=True)
        self.optimizer = optim.Adam([self.xhr_rec], lr=self.init_lr_h)

        self.dim_input = 32
        self.dim_latent_enc = 512
        self.dim_latent_dec=512
        self.dim_noise=32

        self.motion_model = LocalHumanDynamicsGRUNoise(in_dim=self.dim_input, 
                                                       h_dim_enc=self.dim_latent_enc,
                                                       h_dim_dec=self.dim_latent_dec,
                                                       eps_dim=self.dim_noise)

        ## checkpoint loading
        self.motion_model_path = './motion_model/epoch-30.ckp'
        motion_ckpt = torch.load(self.motion_model_path)
        self.motion_model.load_state_dict(motion_ckpt['model_state_dict'])
        self.motion_model.to(self.device)


        ## latent variable init
        self.motion_model_h_enc = torch.zeros([self.batch_size, 1, self.dim_latent_enc ], 
                                dtype=torch.float32).cuda()

        self.motion_model_h_dec = torch.zeros([self.batch_size, 1, self.dim_latent_dec ], 
                                dtype=torch.float32).cuda()




    def cal_loss(self, xhr):
        ### reconstruction loss
        loss_rec = self.weight_loss_rec*F.l1_loss(xhr, self.xhr_rec)
        xh_rec = convert_to_3D_rot(self.xhr_rec)
        # print(xh_rec[:,0:6])
        ### vposer loss
        vposer_pose = xh_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)
        return loss_rec, loss_vposer, 0,0


    def moton_smoothing_loss(self, pred_pose):
        xh_rec = convert_to_3D_rot(self.xhr_rec)
        vposer_pose = xh_rec[:,16:48]
        loss_smoothing = F.l1_loss(vposer_pose, pred_pose)

        return loss_smoothing

    def smoothing_loss(self, xh_prev):
        # xh_rec = convert_to_3D_rot(self.xhr_rec)
        # xh_prev = convert_to_3D_rot(xh_prev)
        # print(self.xhr_rec[:,0:12])
        # print(xh_prev[:,6:48])
        # print(xh_rec[:,48:])
        # print(xh_rec[:,72:])
        loss_smoothing = F.l1_loss(xh_prev[:,0:75], self.xhr_rec[:,0:75])
        return loss_smoothing

    def fitting(self, input_data_file):

        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)
        # initialize
        xh= HumanCVAE.body_params_parse_fitting(body_param_input)
        xhr = convert_to_6D_rot(xh)
        self.xhr_rec.data = xhr.clone()
        for ii in range(self.num_iter):

            self.optimizer.zero_grad()
            loss_rec, loss_vposer,_,_ = self.cal_loss(xhr)
            loss = loss_rec + loss_vposer

            if self.verbose:
                print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}'.format(
                                    ii, loss_rec.item(), loss_vposer.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()
        print('[INFO][fitting] fitting finish, returning optimal value')
        xh_rec =  convert_to_3D_rot(self.xhr_rec)

        return xh_rec


    def fitting_smoothing(self, input_data_file, xh_prev):

        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)

        # initialize
        xh= HumanCVAE.body_params_parse_fitting(body_param_input)
        xhr = convert_to_6D_rot(xh)
        xh_prev = convert_to_6D_rot(xh_prev)
        self.xhr_rec.data = xhr.clone()

        for ii in range(self.num_iter):

            self.optimizer.zero_grad()
            loss_rec, loss_vposer,_,_ = self.cal_loss(xhr)
            loss_smoothing = self.smoothing_loss(xh_prev)
            loss = loss_rec+loss_vposer+loss_smoothing*1.5
            # if self.verbose:
            # print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, total_loss={:f}'.format(
            #                         ii, loss_rec.item(), loss_vposer.item(), 
            #                         loss_smoothing.item(), loss.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()
        print('[INFO][fitting] fitting finish, returning optimal value')
        xh_rec =  convert_to_3D_rot(self.xhr_rec)
        # print(xh_rec[:,0:6])
        return xh_rec

    def fitting_motion_smoothing(self, input_data_file, xh_prev):

        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)

        # initialize
        xh= HumanCVAE.body_params_parse_fitting(body_param_input)
        xhr = convert_to_6D_rot(xh)
        self.xhr_rec.data = xhr.clone()


        pose_prev = xh_prev[:,16:48].unsqueeze(-1).detach() # note that the 10-dim beta is not in xhr
        pose_pred, h_enc_tmp, h_dec_tmp = self.motion_model.forward_seq(pose_prev, seq_length=1, 
                                                                    h_enc = self.motion_model_h_enc,
                                                                    h_dec = self.motion_model_h_dec)
        self.motion_model_h_enc = h_enc_tmp.clone()
        self.motion_model_h_dec = h_dec_tmp.clone()
        pose_pred = pose_pred[:,:,-1].detach()

        for ii in range(self.num_iter):

            self.optimizer.zero_grad()
            loss_rec, loss_vposer,_,_ = self.cal_loss(xhr)
            loss_motion = self.moton_smoothing_loss(pose_pred)
            loss = loss_rec + loss_vposer + loss_motion
            # print(loss)
            # print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_motion={:f}, total_loss={:f}'.format(
            #                         ii, loss_rec.item(), loss_vposer.item(), 
            #                         loss_motion.item(), loss.item()) )
            loss.backward(retain_graph=True)
            self.optimizer.step()
        print('[INFO][fitting] fitting finish, returning optimal value')
        xh_rec =  convert_to_3D_rot(self.xhr_rec)

        return xh_rec


    def save_result(self, xh_rec, output_data_file):

        dirname = os.path.dirname(output_data_file)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        body_param_list = HumanCVAE.body_params_encapsulate(xh_rec)
        print('[INFO] save results to: '+output_data_file)
        for ii, body_param in enumerate(body_param_list):
            # print(body_param['transl'])
            # print(body_param['global_orient'])
            # print()
            # body_param['cam_ext'] = self.cam_ext.detach().cpu().numpy()
            # body_param['cam_int'] = self.cam_int.detach().cpu().numpy()
            outfile = open(output_data_file, 'wb')
            pickle.dump(body_param, outfile)
            outfile.close()


if __name__=='__main__':


    gen_path = sys.argv[1]
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
                'init_lr_h': 0.1,
                'num_iter': 50,
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
    fop = FittingOP(fittingconfig, lossconfig)

    body_gen_list = sorted(glob.glob(os.path.join(gen_path, '*.pkl')))

    freq=1

    for ii in range(1,len(body_gen_list),freq):

        if ii ==1:
            input_data_file = os.path.join(gen_path,'{:04d}.pkl'.format(ii))
            output_data_file=os.path.join(fit_path,'motion_smoothing','{:04d}.pkl'.format(ii))
            xh_rec = fop.fitting(input_data_file)
            fop.save_result(xh_rec, output_data_file)
            xh_prev = xh_rec.detach()
        else:
            input_data_file = os.path.join(gen_path,'{:04d}.pkl'.format(ii))
            output_data_file=os.path.join(fit_path,'motion_smoothing','{:04d}.pkl'.format(ii))
            xh_rec = fop.fitting_smoothing(input_data_file,xh_prev)
            # xh_rec = fop.fitting(input_data_file)
            fop.save_result(xh_rec, output_data_file)
            xh_prev = xh_rec.detach()     
            # break

