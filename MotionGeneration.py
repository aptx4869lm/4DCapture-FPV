from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.utils.clip_grad import *
import copy
import numpy as np
from torch.nn.parameter import Parameter
import os, sys, glob
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import torch.jit as jit
import re
from bisect import bisect_right
import torchgeometry as tgm

from human_body_prior.tools.model_loader import load_vposer
import time



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




class LocalHumanDynamicsGRUNoise(nn.Module):
    '''
    This module tries to model the human motion dynamics, with process noise 
    The human body is represented in the latent space of VPoser, i.e. 32-dim
    The advantage of VPoser is to regularize the generated body pose is natural.
    '''

    def __init__(self, in_dim, h_dim_enc=1024, h_dim_dec=1024, eps_dim=32):
        super(LocalHumanDynamicsGRUNoise, self).__init__()
        
        self.rnn_enc = nn.GRU(input_size=in_dim, 
                              hidden_size=h_dim_enc,
                              num_layers = 1, 
                              batch_first=True,
                              bidirectional=False)
        

        self.enc_mu = nn.Linear(in_features=h_dim_enc, out_features=eps_dim, bias=True)
        self.enc_logvar = nn.Linear(in_features=h_dim_enc, out_features=eps_dim, bias=True)


        self.rnn_dec = nn.GRU(input_size=eps_dim, 
                              hidden_size=h_dim_dec,
                              num_layers = 1, 
                              batch_first=True,
                              bidirectional=False)

        self.out_lin_proj = nn.Linear(in_features=h_dim_dec, 
                                      out_features=in_dim, 
                                      bias=True)


        self.in_dim = in_dim
        self.h_dim_enc = h_dim_enc
        self.h_dim_dec = h_dim_dec
        self.eps_dim = eps_dim
        


    def forward(self, x):
        '''
        residual GRU, see [On human motion prediction using recurrent neural networks]
        x_{t+1} = x_t + f(X_{1:t})

        input: x with shape [batch, feature, time]. Or [x1, x2, x3]
        output: x_pred with shape [batch, feature, time]. Or [x2_pred, x3_pred, x4_pred]
        Note: To use the sampling-based loss, some random frames of the input x are 
              predicted results. Implement this in trainop. 
        '''

        # (1) RNN encoder
        ## - note rnn takes [batch, time, feature] as input
        h_enc,_ = self.rnn_enc(x.permute(0,2,1))
        

        # (2) re-parameterization trick, q(z|x)
        mu_z = self.enc_mu(h_enc)
        logvar_z = self.enc_logvar(h_enc)
        q_z = torch.distributions.normal.Normal(mu_z, 
                                                torch.exp(0.5*logvar_z)
                                                )
        z_sample = q_z.rsample()
        

        # (3) RNN + Linear decoder
        h_dec, _ = self.rnn_dec(z_sample)
        h_dec = self.out_lin_proj(h_dec)
        
        # (4) residual connection
        x_pred = h_dec.permute(0,2,1) + x

        return x_pred, q_z



    def forward_seq(self, x0, seq_length=100, time_window=30,
                    h_enc=None, h_dec=None):
        '''
        Given an initial condition x0, we can recursively predict states in future.
        input: x0 with shape [batch, feature, 1], just one static frame of pose
        '''

        x = x0
        if h_enc is None:
            h_enc = torch.zeros([x.shape[0],1, self.h_dim_enc ], 
                                dtype=torch.float32).cuda()

        if h_dec is None:
            h_dec = torch.zeros([x.shape[0],1, self.h_dim_dec ], 
                                dtype=torch.float32).cuda()


        for _ in range(0, seq_length):
            
            h_enc_tmp,_ = self.rnn_enc(x[:,:,-1:].permute(0,2,1), h_enc[:,-1:,:])

            mu_z = self.enc_mu(h_enc_tmp)
            logvar_z = self.enc_logvar(h_enc_tmp)
            q_z = torch.distributions.normal.Normal(mu_z, 
                                                    torch.exp(0.5*logvar_z)
                                                    )
            z_sample = q_z.rsample()

            h_dec_tmp, _ = self.rnn_dec(z_sample, h_dec[:,-1:,:])
            x_vec = self.out_lin_proj(h_dec_tmp)

            x_pred = x_vec.permute(0,2,1) + x[:,:,-1:]

            # only increase the sequence by 1 each time
            x = torch.cat([x, x_pred], dim=-1)
            h_enc = torch.cat([h_enc, h_enc_tmp], dim=1)
            h_dec = torch.cat([h_dec, h_dec_tmp], dim=1)

        return x, h_enc, h_dec



class TrainOP:
    def __init__(self, modelconfig, lossconfig, trainconfig):
        
        # argument parsing. 
        for key, val in modelconfig.items():
            setattr(self, key, val)

        for key, val in lossconfig.items():
            setattr(self, key, val)

        for key, val in trainconfig.items():
            setattr(self, key, val)


        # build up the model
        self.model = LocalHumanDynamicsGRUNoise(in_dim=self.dim_input, 
                                                h_dim_enc=self.dim_latent_enc,
                                                h_dim_dec=self.dim_latent_dec,
                                                eps_dim=self.dim_noise)
                                                
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')
 

    def rot2vposer(self, x_local):
        '''
        input x: the amass 66D body joint rotation, in which the first 3 are
                 global rotation. 
        We only consider local rotation, and change it to the vposer latent space
        '''

        n_frames = x_local.shape[-1]
        x_local_flatten = x_local.permute([0,2,1]).reshape([-1, 63])
        x_vposer_Gaussian = self.vposer.encode(x_local_flatten) # [b*T, 32]
        x_vposer = x_vposer_Gaussian.loc.reshape([-1, n_frames, 32]).permute([0,2,1])

        return x_vposer


    def rotaa2mat(self, x):
        '''
        input x: the amass 63D local body joint rotation. 
        We only consider local rotation, and change it to the vposer latent space
        '''

        n_frames = x.shape[-1]
        x_aa = x.permute([0,2,1]).reshape([-1, 1, 21, 3])
        x_matrot_9d = self.vposer.aa2matrot(x_aa)[:,0,:,:] # [N, n_joints, 9]
        # x_matrot_6d = x_matrot_9d[:,:,:6].view([-1, n_frames, 21*6]).permute([0,2,1])
        x_matrot_9d = x_matrot_9d.view([-1, n_frames, 21*9]).permute([0,2,1])

        return x_matrot_9d


    def rotmat2aa(self, x):
        '''
        input x: the amass 63D body joint rotation. 
        We only consider local rotation, and change it to the vposer latent space
        '''

        n_frames = x.shape[-1]
        x_9d = x.permute([0,2,1]).view([-1, 1, 21, 9])
        x_3d = self.vposer.matrot2aa(x_9d)[:,0,:,:].reshape([-1,n_frames, 63]).permute([0,2,1])

        return x_3d


    def cal_loss(self, x, ep):

        ### map the joint rotation to vposer latent space
        x_vp = self.rot2vposer(x).detach()
        x_9d = self.rotaa2mat(x).detach()


        ### forward pass of the model
        x_pred_vp, q_z = self.model(x_vp[:,:,:-1]) # x_pred spans t=[2,3,...,T]
        

        ### a simpler version of the sampling based method
        if ep < 0.25* self.num_epochs:
            
            ### decode the x_pred_vp to joint rotation
            x_pred = self.vposer.decode(x_pred_vp.permute([0,2,1]).reshape([-1, 32]),
                                            output_type='matrot') #[-1, 1, n_joints, 9]
            x_pred = x_pred.view([-1, x.shape[-1]-1, 21*9]).permute([0,2,1])
        
            loss_rec = F.l1_loss(x_9d[:,:,1:], x_pred)
            loss_tr = F.l1_loss( x_pred[:,:,1:]-x_pred[:,:,:-1],
                             x_9d[:,:,2:]-x_9d[:,:,1:-1])

            # kl divergence
            mu = q_z.loc
            sigma = q_z.scale
            loss_kl = torch.mean(mu**2 + sigma**2 - 1.0 - torch.log(sigma**2))

            # VPoser
            loss_vp = torch.mean( x_pred_vp**2)

        elif ep < 0.5*self.num_epochs:
            # randomly use the predicted motion as input
            n_frames = x.shape[-1]-1
            x_pred_vp = x_pred_vp.detach()
            mm = torch.tensor(np.random.choice([0,1], 
                              size=[self.batch_size, 1, n_frames-1]),
                              dtype=torch.float32,
                              device=self.device   )
            x_input = x_vp[:,:,:-1]
            x_input2 = x_input.clone().detach()
            x_input2[:,:,1:] = x_input[:,:,1:]*mm + x_pred_vp[:,:,:-1]*(1-mm)
            
            x_pred2_vp, q_z2 = self.model(x_input2.detach())
            x_pred2 = self.vposer.decode(x_pred2_vp.permute([0,2,1]).reshape([-1, 32]),
                                            output_type='matrot') #[-1, 1, n_joints, 9]
            x_pred2 = x_pred2.view([-1, x.shape[-1]-1, 21*9]).permute([0,2,1])
        
            loss_rec = F.l1_loss(x_9d[:,:,1:], x_pred2)
            loss_tr = F.l1_loss( x_pred2[:,:,1:]-x_pred2[:,:,:-1],
                             x_9d[:,:,2:]-x_9d[:,:,1:-1])

            # kl divergence
            mu = q_z2.loc
            sigma = q_z2.scale
            loss_kl = torch.mean(mu**2 + sigma**2 - 1.0 - torch.log(sigma**2))

            # VPoser
            loss_vp = torch.mean( x_pred2_vp**2 )
        
        else:
            # randomly use the predicted motion as input
            n_frames = x.shape[-1]-1
            x_pred_vp = x_pred_vp.detach()
            
            x_input = x_vp[:,:,:-1]
            x_input2 = x_input.clone().detach()
            x_input2[:,:,1:] = x_pred_vp[:,:,:-1]
            
            x_pred2_vp, q_z2 = self.model(x_input2.detach())
            x_pred2 = self.vposer.decode(x_pred2_vp.permute([0,2,1]).reshape([-1, 32]),
                                            output_type='matrot') #[-1, 1, n_joints, 9]
            x_pred2 = x_pred2.view([-1, x.shape[-1]-1, 21*9]).permute([0,2,1])
        
            loss_rec = F.l1_loss(x_9d[:,:,1:], x_pred2)
            loss_tr = F.l1_loss( x_pred2[:,:,1:]-x_pred2[:,:,:-1],
                             x_9d[:,:,2:]-x_9d[:,:,1:-1])

            # kl divergence
            mu = q_z2.loc
            sigma = q_z2.scale
            loss_kl = torch.mean(mu**2 + sigma**2 - 1.0 - torch.log(sigma**2))

            # VPoser
            loss_vp = torch.mean( x_pred2_vp**2 )


        ##** to visualize the input data
        # motion_seq = self.rotmat2aa(x_9d)
        # print(motion_seq.shape)
        # mo_seq_np = motion_seq.detach().to('cpu').numpy()
        # np.savez('input.npz', seq = mo_seq_np)

        # motion_seq1 = self.rotmat2aa(x_pred2)
        # print(motion_seq1.shape)
        # mo_seq_np1 = motion_seq1.detach().to('cpu').numpy()
        # np.savez('input_rec.npz', seq = mo_seq_np1)
        ##** to visualize the input data

        weight_kl = self.weight_kl
        if self.anealing_kl:
           weight_kl = min( ( float(ep) / (0.75*self.num_epochs) )**2, 1.0) * self.weight_kl

        # add them together
        loss = (self.weight_rec*loss_rec 
                + self.weight_temporal*loss_tr
                + weight_kl * loss_kl
                + self.weight_vp*loss_vp)

            
        return loss, loss_rec, loss_tr, loss_kl, loss_vp
        


    def train(self, batch_gen):
        self.model.train()
        self.model.to(self.device)
        self.vposer.eval()
        self.vposer.to(self.device)


        starting_epoch = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.finetunemodel is not None:
            checkpoint = torch.load(self.finetunemodel)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --fine tuning from {}'.format(self.finetunemodel))

        if self.resume_training:
            ckp_list = sorted(glob.glob(os.path.join(self.save_dir,'epoch-*.ckp')),
                                key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                starting_epoch = checkpoint['epoch']
                print('[INFO] --resuming training from {}'.format(ckp_list[-1]))


        for epoch in range(starting_epoch, self.num_epochs):

            epoch_loss = 0
            epoch_loss_rec = 0
            epoch_loss_tr = 0
            epoch_loss_kl = 0
            epoch_loss_vp = 0
            

            stime = time.time()
            while batch_gen.has_next_rec():

                # generate a batch of motion sequence
                batch_input = batch_gen.next_rec_as_a_batch()
                if batch_input is None:
                    continue

                batch_input_pose = batch_input[:,3:66,:].cuda()

                # #** to visualize the input data
                # motion_seq_input = self.vposer.decode(batch_input_vp[0].permute([1,0]),
                #                                 output_type='aa').reshape([-1, 1, 63])

                # motion_seq = motion_seq_input.permute([1,2,0])
                # print(motion_seq.shape)
                # mo_seq_np = motion_seq.detach().to('cpu').numpy()
                # np.savez('input.npz', seq = mo_seq_np)
                # #** to visualize the input data
                
                optimizer.zero_grad()
            
                # calculate noise
                [loss, 
                 loss_rec,
                 loss_tr,
                 loss_kl,
                 loss_vp] = self.cal_loss(batch_input_pose, epoch)

                epoch_loss += loss.item()
                epoch_loss_rec += loss_rec.item()
                epoch_loss_tr += loss_tr.item()
                epoch_loss_kl += loss_kl.item()
                epoch_loss_vp += loss_vp.item()

                loss.backward(retain_graph=False)
                clip_grad_norm_(self.model.parameters(), max_norm=0.001)
                optimizer.step()
                

            batch_gen.reset()


            if ((1+epoch) % self.saving_per_X_ep==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, self.save_dir + "/epoch-" + str(epoch + 1) + ".ckp") 

            if self.verbose:
                print("[epoch {:d}]: loss={:f}, loss_rec={:f}, loss_tr={:f}, loss_kl={:f}, loss_vp={:f}, time={:f}sec"
                            .format(epoch + 1, 
                                    epoch_loss / len(batch_gen.rec_list), 
                                    epoch_loss_rec / len(batch_gen.rec_list),
                                    epoch_loss_tr / len(batch_gen.rec_list),
                                    epoch_loss_kl / len(batch_gen.rec_list),
                                    epoch_loss_vp / len(batch_gen.rec_list),
                                    time.time()-stime
                                    ))

        if self.verbose:
            print('[INFO]: Training completes!')
            print()



class TestOP:
    def __init__(self, modelconfig, testconfig):
        

        # argument parsing. 

        for key, val in modelconfig.items():
            setattr(self, key, val)

        for key, val in testconfig.items():
            setattr(self, key, val)
    

        if not os.path.exists(self.model_ckpt):
            print('[ERROR]: no model was trained. Program terminates.')
            sys.exit(-1)

        if not os.path.exists(self.results_dir):
            print('[INFO] -- create the results_dir')
            os.mkdir(self.results_dir)


        # build up the model
        self.model = LocalHumanDynamicsGRUNoise(in_dim=self.dim_input, 
                                                h_dim_enc=self.dim_latent_enc,
                                                h_dim_dec=self.dim_latent_dec,
                                                eps_dim=self.dim_noise)
                                                        
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')



    def generate(self, motion_len_list):
        '''
        Given a set of (initial static pose, an expected motion seq length),
        this script outputs the LOCAL human motion...
        '''

        self.model.eval()
        self.vposer.eval()

        with torch.no_grad():
            self.model.to(self.device)
            self.vposer.to(self.device)

            checkpoint = torch.load(self.model_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])            

            ## sequence intialization
            ## first, we randomly sample poses from VPoser
            n_seq = len(motion_len_list)
            pose_ini = torch.randn(n_seq, self.dim_input,1).to('cuda')
        

            # generate motion sequence
            motion_seq_list = []
            motion_seq_vp_list = []

            for idx, motion_len in enumerate(motion_len_list):
        
                ## first, generate motion in the Vpose latent space
                motion_seq_vposer,_,_ = self.model.forward_seq( pose_ini[idx:idx+1, :,:], 
                                                           seq_length=motion_len).squeeze()
                mo_seq_vp_np = motion_seq_vposer.detach().to('cpu').numpy()

                ## second, recover joint rotations via vposer decoder
                ## note that 'aa' means axis angular rotation
                motion_seq = self.vposer.decode(motion_seq_vposer.permute([1,0]),
                                                output_type='aa').reshape([-1, 1, 63])
                
                motion_seq = motion_seq.permute([1,2,0])
                print(motion_seq.shape)
                mo_seq_np = motion_seq.detach().to('cpu').numpy()

                ## third, put all motion sequences into the list
                motion_seq_list.append(mo_seq_np)
                motion_seq_vp_list.append(mo_seq_vp_np)


            # save motion sequence
            prefix = 'motion_gen_'
            filelist = sorted(glob.glob(os.path.join(self.results_dir, prefix+'*')))
                              

            if len(filelist)!=0:
                filenewestidx = int(filelist[-1][-9:-4])
            else:
                filenewestidx=0

            for idx in range(len(motion_seq_list)):
                filename = prefix+'{:05d}.npz'.format(filenewestidx+1+idx)
                print('[INFO] -- generate: '+filename)
                np.savez(os.path.join(self.results_dir, filename),
                                     seq=motion_seq_list[idx], 
                                     seq_vp=motion_seq_vp_list[idx])

            print('[INFO]: DONE! Go to the visualization script to check.')
            print()

