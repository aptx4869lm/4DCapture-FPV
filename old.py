
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
import chamfer_pytorch.dist_chamfer as ext

from cvae import HumanCVAE, HumanCVAE5, ContinousRotReprDecoder
from batch_gen_hdf5 import BatchGeneratorWithSceneMesh




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




        ## read scene sdf
        with open(self.scene_sdf_path+'.json') as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'])
                grid_max = np.array(sdf_data['max'])
                grid_dim = sdf_data['dim']
        sdf = np.load(self.scene_sdf_path + '_sdf.npy').reshape(grid_dim, grid_dim, grid_dim)

        self.s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_sdf_batch = self.s_sdf_batch.repeat(self.batch_size,1,1,1)

        ## read scene vertices
        scene_o3d = o3d.io.read_triangle_mesh(self.scene_verts_path)
        scene_verts = np.asarray(scene_o3d.vertices)
        self.s_verts_batch = torch.tensor(scene_verts, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_verts_batch = self.s_verts_batch.repeat(self.batch_size, 1,1)




    def cal_loss(self, xhr, cam_ext):

        
        ### reconstruction loss
        loss_rec = self.weight_loss_rec*F.l1_loss(xhr, self.xhr_rec)*100000
        xh_rec = convert_to_3D_rot(self.xhr_rec)

        ### vposer loss
        vposer_pose = xh_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)


        ### contact loss
        body_param_rec = HumanCVAE.body_params_encapsulate_batch(xh_rec)
        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)
 
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
        body_verts_batch = verts_transform(body_verts_batch, cam_ext)

        vid, fid = get_contact_id(body_segments_folder='/is/cluster/work/yzhang/PROX/body_segments',
                                contact_body_parts=self.contact_part)
        body_verts_contact_batch = body_verts_batch[:, vid, :]

        dist_chamfer_contact = ext.chamferDist()
        contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch.contiguous(), 
                                                self.s_verts_batch.contiguous())

        loss_contact = self.weight_contact * torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0))  



        ### sdf collision loss
        s_grid_min_batch = self.s_grid_min_batch.unsqueeze(1)
        s_grid_max_batch = self.s_grid_max_batch.unsqueeze(1)

        norm_verts_batch = (body_verts_batch - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) *2 -1
        n_verts = norm_verts_batch.shape[1]
        body_sdf_batch = F.grid_sample(self.s_sdf_batch.unsqueeze(1), 
                                        norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
                                        padding_mode='border')


        # if there are no penetrating vertices then set sdf_penetration_loss = 0
        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            loss_sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()

        loss_collision = self.weight_collision*loss_sdf_pene

        return loss_rec, loss_vposer, loss_contact, loss_collision




    def fitting(self, input_data_file):

        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)

        # initialize
        xh= HumanCVAE.body_params_parse_fitting(body_param_input)
        xhr = convert_to_6D_rot(xh)
        self.xhr_rec.data = xhr.clone()

        
        for ii in range(self.num_iter):

            self.optimizer.zero_grad()

            loss_rec, loss_vposer, loss_contact, loss_collision = self.cal_loss(xhr, self.cam_ext)
            loss = loss_rec + loss_vposer + loss_contact + loss_collision
            if self.verbose:
                print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, l_contact={:f}, l_collision={:f}'.format(
                                        ii, loss_rec.item(), loss_vposer.item(), 
                                        loss_contact.item(), loss_collision.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()


        print('[INFO][fitting] fitting finish, returning optimal value')


        xh_rec =  convert_to_3D_rot(self.xhr_rec)

        return xh_rec


    # def fitting(self, xhr_batch, cam_ext_batch, cam_int_batch):

    #     # with open(input_data_file, 'rb') as f:
    #     #     body_param_input = pickle.load(f)


    #     # xh, self.cam_ext, self.cam_int= HumanCVAE.body_params_parse_fitting(body_param_input)
    #     # xhr = convert_to_6D_rot(xh)
    #     self.xhr_rec.data = xhr_batch.clone()

        
    #     for ii in range(self.num_iter):

    #         self.optimizer.zero_grad()

    #         loss_rec, loss_vposer, loss_contact, loss_collision = self.cal_loss(xhr_batch, cam_ext_batch)
    #         loss = loss_rec + loss_vposer + loss_contact + loss_collision
    #         if self.verbose:
    #             print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, l_contact={:f}, l_collision={:f}'.format(
    #                                     ii, loss_rec.item(), loss_vposer.item(), 
    #                                     loss_contact.item(), loss_collision.item()) )

    #         loss.backward(retain_graph=True)
    #         self.optimizer.step()


    #     print('[INFO][fitting] fitting finish, returning optimal value')


    #     xh_rec_batch =  convert_to_3D_rot(self.xhr_rec)

    #     return xh_rec_batch




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
            body_param['cam_ext'] = self.cam_ext.detach().cpu().numpy()
            body_param['cam_int'] = self.cam_int.detach().cpu().numpy()
            outfile = open(output_data_file, 'wb')
            pickle.dump(body_param, outfile)
            outfile.close()


if __name__=='__main__':


    gen_path = sys.argv[1]
    fit_path = sys.argv[2]

    scene_test_list = ['MPH16', 'MPH1Library','N0SittingBooth', 'N3OpenArea']

    for scenename in scene_test_list:
        fittingconfig={
                    # 'input_data_file': input_data_file,
                    # 'output_data_file': os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii)),
                    # 'output_data_file': '/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_baseline_postproc/realcams/'+scenename+'/body_gen_{:06d}.pkl'.format(ii),
                    'scene_verts_path': '/is/cluster/work/yzhang/PROX/scenes_downsampled/'+scenename+'.ply',
                    'scene_sdf_path': '/is/cluster/work/yzhang/PROX/scenes_sdf/'+scenename,
                    'human_model_path': '/is/ps2/yzhang/body_models/VPoser',
                    'vposer_ckpt_path': '/is/ps2/yzhang/body_models/VPoser/vposer_v1_0',
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
            'weight_loss_vposer':0.01,
            'weight_contact': 0.1,
            'weight_collision' : 0.5
        }
        fop = FittingOP(fittingconfig, lossconfig)


        # batch_size = fittingconfig['batch_size']
        
        # ii = 0
        # while ii < 1200:

        #     xhr_list = []
        #     cam_ext_list = []
        #     cam_int_list = []
        #     for jj in range(ii, ii+batch_size):
        #         input_data_file = os.path.join(gen_path,scenename+'/body_gen_{:06d}.pkl'.format(jj))

        #         with open(input_data_file, 'rb') as f:
        #             body_param_input = pickle.load(f)

        #         xh, cam_ext, cam_int= HumanCVAE.body_params_parse_fitting(body_param_input)
        #         xhr = convert_to_6D_rot(xh)
        #         xhr_list.append(xhr)
        #         cam_ext_list.append(cam_ext)
        #         cam_int_list.append(cam_int)

        #     xhr_batch = torch.cat(xhr_list, dim=0)
        #     cam_ext_batch = torch.cat(cam_ext_list, dim=0)
        #     cam_int_batch = torch.cat(cam_int_list, dim=0)

        #     xh_rec_batch = fop.fitting(xhr_batch, cam_ext_batch, cam_int_batch)

        #     for jj in range(ii, ii+batch_size):
        #         output_data_file = os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(jj))
        #         xh_rec = xh_rec_batch[jj-ii : jj-ii+1, :]             
        #         cam_ext = cam_ext_batch[jj-ii : jj-ii+1, :]             
        #         cam_int = cam_int_batch[jj-ii : jj-ii+1, :]             
        #         fop.save_result(xh_rec, cam_ext, cam_int, output_data_file)

        #     ii = ii+batch_size




        for ii in range(1200):

            input_data_file = os.path.join(gen_path,scenename+'/body_gen_{:06d}.pkl'.format(ii))
            if not os.path.exists(input_data_file):
                continue

            output_data_file=os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii))
            
            if os.path.exists(output_data_file):
                continue

            xh_rec = fop.fitting(input_data_file)
            fop.save_result(xh_rec, output_data_file)

