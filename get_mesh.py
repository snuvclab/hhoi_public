import numpy as np
import torch
import smplx
import trimesh
import argparse
from tqdm import tqdm
import os
import pickle 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--inference_dir", type=str)
    args = parser.parse_args()

    start_gray = 200 
    end_gray = 50

    # smplx dir should be in {model_path}
    smplx_model = smplx.create(model_path=".", model_type='smplx', num_expression_coeffs = 100).to("cuda")
    smplx_model.eval()
    beta = torch.zeros((1, 10)).to("cuda")

    save_dir = os.path.join(args.inference_dir, "mesh_outputs")
    os.makedirs(save_dir, exist_ok=True)

    model_output = np.load(f"{args.inference_dir}/inference.npz", allow_pickle=True)
    transform = model_output['transform']
    model_type = model_output['model_type'].item()
    
    for i in tqdm(range(len(transform))):
        if model_type == 'hoi':
            p_left_orient = torch.tensor(transform[i, :3]).reshape((1, -1)).float().to("cuda")
            p_left_transl = torch.tensor(transform[i, 3:6]).reshape((1, -1)).float().to("cuda")
            p_left_scale = torch.tensor(transform[i, 6:7]).float().to("cuda")
            p_left_pose = torch.tensor(transform[i, 7:]).reshape((1, -1)).float().to("cuda")
            
            p_left_vert = smplx_model(global_orient = p_left_orient, betas = beta, body_pose = p_left_pose).vertices
            p_left_vert = p_left_vert * p_left_scale + p_left_transl

            human = trimesh.Trimesh(vertices=p_left_vert.detach().cpu().squeeze(), faces=smplx_model.faces)
            asset = trimesh.load(f"{args.data_dir}/asset.obj")

            full = human + asset
            full.export(f"{save_dir}/%05d.obj"%i)
        elif model_type == 'hhi':
            p_left_pose = torch.tensor(transform[i, :63]).reshape((1, -1)).float().to("cuda")
            p_right_orient = torch.tensor(transform[i, 63:66]).reshape((1, -1)).float().to("cuda")
            p_right_transl = torch.tensor(transform[i, 66:69]).reshape((1, -1)).float().to("cuda")
            p_right_pose = torch.tensor(transform[i, 69:]).reshape((1, -1)).float().to("cuda")

            p_left_vert = smplx_model(global_orient = torch.zeros((1, 3)).to("cuda"), betas = beta, body_pose = p_left_pose, transl = torch.zeros((1, 3)).to("cuda")).vertices.detach().cpu().squeeze()
            p_right_vert = smplx_model(global_orient = p_right_orient, betas = beta, body_pose = p_right_pose, transl = p_right_transl).vertices.detach().cpu().squeeze()

            left_human = trimesh.Trimesh(vertices = p_left_vert, faces = smplx_model.faces)
            right_human = trimesh.Trimesh(vertices = p_right_vert, faces = smplx_model.faces)

            full = left_human + right_human
            full.export(f"{save_dir}/%05d.obj"%i)
        elif model_type == 'hhoi':
            asset = trimesh.load(f"{args.data_dir}/asset.obj")
            meshes = [asset]
            #full = trimesh.load(f"{args.data_dir}/asset.obj")

            num_human = transform.shape[1]
            for h_idx in range(num_human):
                p_orient = torch.tensor(transform[i, h_idx, :3]).reshape((1, -1)).float().to("cuda")
                p_transl = torch.tensor(transform[i, h_idx, 3:6]).reshape((1, -1)).float().to("cuda")
                p_scale = torch.tensor(transform[i, h_idx, 6:7]).float().to("cuda")
                p_pose = torch.tensor(transform[i, h_idx, 7:]).reshape((1, -1)).float().to("cuda")
                
                with torch.no_grad():
                    p_vert = smplx_model(global_orient = p_orient, betas = beta, body_pose = p_pose).vertices
                    p_vert = p_vert[0] * p_scale + p_transl
    

                t = h_idx / (num_human - 1) if num_human > 1 else 0.0
                gray = int((1 - t) * start_gray + t * end_gray)
                color = np.array([gray, gray, gray, 255], dtype=np.uint8)
                colors = np.tile(color[None, :], (p_vert.shape[0], 1))  # (N,4)
                human = trimesh.Trimesh(vertices=p_vert.detach().cpu().squeeze(), faces=smplx_model.faces, vertex_colors=colors, process=False)
                human.visual.vertex_colors = colors

                #human = trimesh.Trimesh(vertices = p_vert.detach().cpu().squeeze(), faces = smplx_model.faces)
                #full += human
                meshes.append(human)

            full = trimesh.util.concatenate(meshes)
            full.export(f"{save_dir}/%05d.obj"%i)

