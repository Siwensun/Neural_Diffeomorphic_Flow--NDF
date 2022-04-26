#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import argparse
import json
import time
import numpy as np
import os
import torch
import plyfile
import sys
from sklearn.neighbors import KDTree
import pyvista as pv
import pyacvd

import deep_sdf
import deep_sdf.workspace as ws


def save_to_ply(verts, verts_warped, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    # store canonical coordinates as rgb color (in float format)
    verts_color = 255 * (0.5 + 0.5 * verts_warped)
    verts_color = verts_color.astype(np.uint8)

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                          verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

def remesh_acvd(ori_verts, ori_faces, cluster_num, divide=False):
    faces_2 = np.ones([ori_faces.shape[0], 4], dtype=int) * 3
    faces_2[:, 1:4] = ori_faces
    surf = pv.PolyData(ori_verts.astype(float), faces_2)

    clus = pyacvd.Clustering(surf)
    if divide:
        # mesh is not dense enough for uniform remeshing
        clus.subdivide(3)
    clus.cluster(cluster_num)
    remesh = clus.create_mesh()

    verts_remesh = np.asarray(remesh.points)
    faces_remesh = remesh.faces.reshape((-1, 4))[:, 1:4]

    # clean mesh
    verts_remesh, faces_remesh = remove_isolated_vertices(verts_remesh, faces_remesh)

    return verts_remesh, faces_remesh

def remove_isolated_vertices(verts, faces):
    indice = np.arange(verts.shape[0])
    indice_f = np.unique(faces)
    isolated_idx = set(indice) - set(indice_f)

    if len(isolated_idx) > 0:
        isolated_idx = sorted(isolated_idx, reverse=True)
        verts = np.delete(verts, isolated_idx, 0)
        for idx in isolated_idx:
            faces[faces > idx] -= 1

    return verts, faces

def get_template_mesh(experiment_directory, saved_model_epoch, num_clusters):
    # load template mesh
    template_filename = os.path.join(experiment_directory,
        ws.training_meshes_subdir,
        str(saved_model_epoch), 'template')

    if os.path.exists(template_filename + f"_{num_clusters}_color_coded.ply"):
        logging.info(f"Loading from {template_filename}_{num_clusters}_color_coded.ply")

        template_remesh = plyfile.PlyData.read(template_filename + f"_{num_clusters}_color_coded.ply")
        template_remesh_v = []
        template_remesh_f = []
        template_remesh_v_colors = []
        for v in template_remesh.elements[0]:
            template_remesh_v.append(np.array((v[0], v[1], v[2])))
        for f in template_remesh.elements[1]:
            f = f[0]
            template_remesh_f.append(np.array([f[0], f[1], f[2]]))
        for v in template_remesh.elements[0]:
            template_remesh_v_colors.append(np.array((v[3], v[4], v[5])))
        template_remesh_v = np.asarray(template_remesh_v)
        template_remesh_f = np.asarray(template_remesh_f)
        template_remesh_v_colors = np.asarray(template_remesh_v_colors) / 255
    else:
        logging.info("Loading from %s.ply" % template_filename)
        template = plyfile.PlyData.read(template_filename + ".ply")
        template_v = [] #template.elements[0]
        template_f = [] #template.elements[1]
        for i in range(template.elements[0].count):
            v = template.elements[0][i]
            template_v.append(np.array((v[0], v[1], v[2])))
        for i in range(template.elements[1].count):
            f = template.elements[1][i][0]
            template_f.append(np.array([f[0], f[1], f[2]]))
        template_v = np.asarray(template_v)
        template_f = np.asarray(template_f)

        template_remesh_v, template_remesh_f = remesh_acvd(template_v, template_f, num_clusters)

        template_remesh_v_colors = 0.5 + 0.5 * template_remesh_v

        save_to_ply(template_remesh_v, 
                    template_remesh_v, 
                    template_remesh_f, 
                    template_filename + f"_{num_clusters}_color_coded.ply")

    return template_remesh_v, template_remesh_f, template_remesh_v_colors

def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def optimze_pts(warper, template_v, xyz_init, latent_vector, loss_fn, num_iterations=500, decreased_by=10, num_samples=30000, lr=5e-3):
    adjust_lr_every = int(num_iterations / 6)
    batch_num = template_v.shape[0] // num_samples + 1

    xyzs = []

    for batch_id in range(batch_num):
        xyz_warped = torch.from_numpy(template_v[num_samples*batch_id:num_samples*(batch_id+1)].astype(np.float32)).cuda()
        latent_inputs = latent_vector.expand(xyz_warped.shape[0], -1).cuda()

        if xyz_init is None:
            xyz = torch.ones(xyz_warped.shape[0], 3).normal_(mean=0, std=0.01).cuda()
        else:
            xyz = torch.from_numpy(xyz_init.astype(np.float32)).cuda()
        xyz.requires_grad = True

        optimizer = torch.optim.Adam([xyz], lr=lr)
        
        for e in range(num_iterations):
            warper.eval()

            adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

            optimizer.zero_grad()

            inputs = torch.cat([latent_inputs, xyz], 1)

            pred_xyz = warper(inputs)[0]

            loss = loss_fn(pred_xyz, xyz_warped)

            loss.backward()
            optimizer.step()

            if e % 100 == 0:
                logging.info(f'{batch_id}\t{e}\t{loss.item()}')

            torch.cuda.empty_cache()

        xyzs.append(xyz)
        del xyz, xyz_warped, optimizer
        
    return torch.cat(xyzs).detach().cpu().numpy()

def ode_back_pts(warper, template_v, latent_vector):
    template_v = torch.from_numpy(template_v.astype(np.float32)).cuda()
    num_samples = template_v.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    inputs = torch.cat([latent_repeat, template_v], 1)
    warped_back = []
    head = 0
    max_batch = 2**17
    while head < num_samples:
        with torch.no_grad():
            warped_back_, _ = warper(inputs[head : min(head + max_batch, num_samples)], invert=True)
        warped_back_ = warped_back_.detach().cpu().numpy()
        warped_back.append(warped_back_)
        head += max_batch
    warped_back = np.concatenate(warped_back, axis=0)

    return warped_back

def mesh_to_topology_correspondence(experiment_directory, 
                                    checkpoint, 
                                    start_id, 
                                    end_id, 
                                    num_clusters,
                                    bp_or_ode='bp',
                                    test_or_train='train'):

    # learning parameters
    num_iterations = 500
    decreased_by = 10
    num_samples = 35000
    loss_l1 = torch.nn.L1Loss()
    lr=5e-3

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    Warper, saved_model_epoch = ws.prepare_warper(specs, experiment_directory, checkpoint)

    Warper.eval()

    if test_or_train == 'train':
        split_file = specs["TrainSplit"]
    else:
        split_file = specs["TestSplit"]

    with open(split_file, "r") as f:
        split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, split)
    
    if test_or_train == 'train':
        latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, saved_model_epoch)
    else:
        latent_vectors = ws.load_optimized_test_latent_vectors(experiment_directory, saved_model_epoch, instance_filenames)
    latent_vectors = latent_vectors.cuda()

    template_remesh_v, template_remesh_f, template_remesh_v_colors = get_template_mesh(experiment_directory, 
                                                                                       saved_model_epoch, 
                                                                                       num_clusters)

    for i, instance_filename in enumerate(instance_filenames):
        if i < start_id:
            continue

        latent_vector = latent_vectors[i]

        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filename).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filename).split("\\")
        instance_name = ".".join(instance_name.split(".")[:-1])

        if test_or_train == 'train':
            mesh_dir = os.path.join(
                experiment_directory,
                ws.training_meshes_subdir,
                str(saved_model_epoch),
                dataset_name,
                class_name,
            )
        else:
            mesh_dir = os.path.join(
                experiment_directory,
                ws.reconstructions_subdir,
                str(saved_model_epoch),
                ws.reconstruction_meshes_subdir,
                dataset_name,
                class_name,
            )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)
        if os.path.exists(mesh_filename+ "_color_coded.ply"):
            logging.info("Loading from %s_color_coded.ply" % mesh_filename)

            mesh = plyfile.PlyData.read(mesh_filename + "_color_coded.ply")
            mesh_v = []
            mesh_f = []
            mesh_v_c = []
            for v in mesh.elements[0]:
                mesh_v.append(np.array((v[0], v[1], v[2])))
            for f in mesh.elements[1]:
                f = f[0]
                mesh_f.append(np.array([f[0], f[1], f[2]]))
            for v in mesh.elements[0]:
                mesh_v_c.append(np.array((v[3], v[4], v[5])))
            mesh_v = np.asarray(mesh_v)
            mesh_f = np.asarray(mesh_f)
            mesh_v_c = np.asarray(mesh_v_c) / 255

            tree = KDTree(mesh_v_c) 
            remesh_ind = tree.query(template_remesh_v_colors, return_distance=False)
            remesh_v = mesh_v[remesh_ind[:, 0], :]

            save_to_ply(remesh_v, 
                        template_remesh_v, 
                        template_remesh_f, 
                        mesh_filename + f"_{num_clusters}_coarse_correspondence.ply")

            
            if bp_or_ode == 'bp':
                xyzs = optimze_pts(Warper, 
                                template_remesh_v, 
                                remesh_v, 
                                latent_vector, 
                                loss_l1, num_iterations, decreased_by, num_samples, lr)
                save_to_ply(xyzs, 
                            template_remesh_v, 
                            template_remesh_f, 
                            mesh_filename + f"_{num_clusters}_fine_correspondence.ply")
            elif bp_or_ode == 'ode':
                start_time = time.time()
                xyzs = ode_back_pts(Warper, 
                                    template_remesh_v, 
                                    latent_vector)
                end_time = time.time()
                print(end_time - start_time)
                save_to_ply(xyzs, 
                            template_remesh_v, 
                            template_remesh_f, 
                            mesh_filename + f"_{num_clusters}_fine_correspondence.ply")


        if i >= end_id:
            break


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--test_or_train",
        "-t",
        dest="test_or_train",
        required=True,
        help="Whether to evaluate training meshes or reconstructed meshes",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--start_id",
        dest="start_id",
        type=int,
        default=0,
        help="start_id.",
    )
    arg_parser.add_argument(
        "--end_id",
        dest="end_id",
        type=int,
        default=20,
        help="end_id.",
    )
    arg_parser.add_argument(
        "--num_clusters",
        dest="num_clusters",
        type=int,
        default=20,
        help="Apply AVCD algorithm to remesh template mesh; Generate specific topology that all shapes should follow"
    )
    arg_parser.add_argument(
        "--bp_or_ode",
        dest="bp_or_ode",
        type=str,
        default='bp',
        help="For DiT, we could use back propogation to get fine mapping results; For DDiT, we could use NODE with reverse time"
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    mesh_to_topology_correspondence(args.experiment_directory, 
                                    args.checkpoint, 
                                    args.start_id, 
                                    args.end_id, 
                                    args.num_clusters, 
                                    args.bp_or_ode,
                                    args.test_or_train)
