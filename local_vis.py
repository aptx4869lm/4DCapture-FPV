
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



import numpy as np


class MeshViewer(object):

    def __init__(self, width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        import trimesh
        import pyrender

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3])
        self.scene.add(pc, pose=camera_pose)

        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                                      viewport_size=(width, height),
                                      cull_faces=False,
                                      run_in_thread=True,
                                      registered_keys=registered_keys)

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces):
        if not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name='body_mesh')

        self.viewer.render_lock.release()


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=True,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func

with FittingMonitor() as monitor:
# print(camera_pose)
    light_nodes = monitor.mv.viewer._create_raymond_lights()
def main(fitting_dir):

    # fitting_dir ='/home/miao/data/rylm/segmented_data/miao_downtown_1-0/body_gen'
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
    i =0
    for body_file in sorted(glob.glob(fitting_dir+'/*.pkl')):
        print('viz frame {}'.format(body_file))
        ## get humam body params
        # if not os.path.exists(filename):
        #     continue

        with open(body_file, 'rb') as f:
            param = pickle.load(f)
            pose_embedding= param['body_pose']
            camera_transl = param['camera_translation']
        # print(param)
        pose_embedding = torch.tensor(pose_embedding, dtype=torch.float32)
        # print(pose_embedding)
        # camera_transl = torch.tensor(camera_transl, dtype=torch.float32).detach().cpu().numpy().squeeze()
        camera_transl = torch.tensor(camera_transl, dtype=torch.float32)
        
        body_pose = vposer.decode(pose_embedding,output_type='aa').view(1, -1) 

        body_dict={}
        for key, val in param.items():
            if key in [
                       'jaw_pose', 'leye_pose','reye_pose','expression']:
                continue

            else:
                body_dict[key]=torch.tensor(param[key], dtype=torch.float32)

        body_dict['body_pose']=body_pose
        # body_dict['transl']=camera_transl
        # print(camera_transl)
        model_output = body_mesh_model(return_verts=True, **body_dict)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        vertices = vertices*2.6951
        import trimesh
        out_mesh = trimesh.Trimesh(vertices, body_mesh_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_transl = camera_transl.detach().cpu().numpy().squeeze()
        camera_transl = camera_transl*2.6951
        camera_transl[0] *= -1.0

        camera_pose = np.eye(4)

        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=692, fy=692,
            cx=640, cy=360)
        scene.add(camera, pose=camera_pose)
        # print(camera_pose)
        # print(body_dict['transl'])
        # print(body_dict['global_orient'])
        # print(body_dict['camera_translation'])
        for node in light_nodes:
            scene.add_node(node)

        r = pyrender.OffscreenRenderer(viewport_width=1280,
                                       viewport_height=720,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

        # img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        imgname= body_file.split('/')[-1].replace('.pkl','').replace('body_gen_','')
        imgname=str(i+1)
        output_dir = fitting_dir.replace('smoothed_body','local_vis')
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(output_dir)
        write_name=output_dir+'/'+imgname.zfill(4)+'.png'
        img_dir = fitting_dir.replace('smoothed_body','images')
        # print(img_dir)
        read_name=img_dir+'/'+imgname.zfill(6)+'.jpg'
        print(read_name)
        input_img = cv2.imread(read_name).astype(np.float32)[:, :, ::-1] / 255.0

        output_img = (color[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * input_img)

        img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        img.save(write_name)
        i = i+1
        # break
            # print(valid_mask)
            # break
        # for key, val in param['body'].items():
        #     if key in ['camera_rotation', 'camera_translation', 
        #                'jaw_pose', 'leye_pose','reye_pose','expression']:
        #         continue

        #     else:
        #         body_dict[key]=torch.tensor(param['body'][key], dtype=torch.float32)

        # smplx_output = body_mesh_model(return_verts=True, **body_dict)
        # body_verts_batch = smplx_output.vertices #[b, 10475,3]
        # body_verts_np = body_verts_batch.detach().cpu().numpy().squeeze()

        # ## read body mesh
        # body.vertices = o3d.utility.Vector3dVector(body_verts_np)
        # body.triangles = o3d.utility.Vector3iVector(body_mesh_model.faces)
        # body.vertex_normals = o3d.utility.Vector3dVector([])
        # body.triangle_normals = o3d.utility.Vector3dVector([])
        # body.compute_vertex_normals()
        # body.transform(trans)
        # vis.update_geometry(body)
        
        # ## update cam for render
        # ctr = vis.get_view_control()
        # cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param = update_cam(cam_param, trans)
        # ctr.convert_from_pinhole_camera_parameters(cam_param)

        # ## capture RGB appearance
        # rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        # cv2.imshow("frame", np.uint8(255*rgb[:,:,[2,1,0]]))
        # renderimgname = os.path.join(outrenderfolder, 'img_{:03d}.png'.format(count))
        # cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        # cv2.waitKey(300)

        # count += 1

if __name__=='__main__':
    fitting_dir = sys.argv[1]
    main(fitting_dir)