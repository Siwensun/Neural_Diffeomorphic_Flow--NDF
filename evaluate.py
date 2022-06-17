#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
from pathos.multiprocessing import ProcessPool as Pool

import deep_sdf
import deep_sdf.workspace as ws


def evaluate_one_instance(dataset, 
                          class_name, 
                          instance_name, 
                          experiment_directory, 
                          checkpoint, 
                          data_dir, 
                          test_or_train='test',
                          correspondence_level=None, 
                          correspondence_pts_num=0):
    logging.debug(
        "evaluating " + os.path.join(dataset, class_name, instance_name)
    )
    if test_or_train == 'test':
        mesh_filename = ws.get_reconstructed_mesh_filename(
            experiment_directory, checkpoint, dataset, class_name, instance_name, correspondence_level, correspondence_pts_num
        )

        logging.debug(
            'reconstructed mesh is "' + mesh_filename + '"'
        )
    else:
        mesh_filename = ws.get_trained_mesh_filename(
            experiment_directory, checkpoint, dataset, class_name, instance_name, correspondence_level, correspondence_pts_num
        )

        logging.debug(
            'trained mesh is "' + mesh_filename + '"'
        )

    if not os.path.isfile(mesh_filename):
        print('[WARNING] Skipping %s as it doesn\'t exists' % mesh_filename)
        return "", 0

    ground_truth_points_samples_filename = os.path.join(
        data_dir,
        "SurfaceSamples",
        dataset,
        class_name,
        instance_name + ".ply",
    )

    logging.debug(
        "ground truth points samples are " + ground_truth_points_samples_filename
    )

    ground_truth_mesh_samples_filename = os.path.join(
        data_dir,
        "MeshSamples",
        dataset,
        class_name,
        instance_name + ".ply",
    )

    logging.debug(
        "ground truth mesh samples are " + ground_truth_mesh_samples_filename
    )

    normalization_params_filename = os.path.join(
        data_dir,
        "NormalizationParameters",
        dataset,
        class_name,
        instance_name + ".npz",
    )

    logging.debug(
        "normalization params are " + normalization_params_filename
    )

    ground_truth_points = trimesh.load(ground_truth_points_samples_filename)
    ground_truth_mesh = trimesh.load(ground_truth_mesh_samples_filename)
    reconstruction = trimesh.load(mesh_filename)

    if os.path.exists(normalization_params_filename):
        normalization_params = np.load(normalization_params_filename)
    else:
        normalization_params = {"offset": 0, "scale": 1}

    metrics = {}
    
    chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
        ground_truth_points,
        reconstruction,
        normalization_params["offset"],
        normalization_params["scale"],
    )
    metrics = {**metrics, **chamfer_dist}

    earthmover_dist = deep_sdf.metrics.emd.compute_trimesh_emd(
        ground_truth_points,
        reconstruction,
        normalization_params["offset"],
        normalization_params["scale"],
    )
    metrics = {**metrics, **earthmover_dist}

    non_manifold = deep_sdf.metrics.non_manifold.calculate_manifoldness(reconstruction)
    metrics = {**metrics, **non_manifold}
    
    normal_consistency = deep_sdf.metrics.normal_consistency.compute_geometric_metrics_points(
        ground_truth_mesh,
        reconstruction
    )
    metrics = {**metrics, **normal_consistency}
    
    for key in metrics:
        logging.debug(f"{key}: {metrics[key]}")

    return os.path.join(dataset, class_name, instance_name), metrics


def evaluate(experiment_directory, checkpoint, data_dir, split_filename, 
             test_or_train='test', correspondence_level=None, correspondence_pts_num = 0):

    with open(split_filename, "r") as f:
        split = json.load(f)

    results = []
    p = Pool(8)
    ds = []
    cn = []
    inn = []
    exd = []
    ckp = []
    dtd = []
    tot = []
    cl = []
    cpn = []

    print('data_preparing')
    for dataset in split:
        for class_name in split[dataset]:
            for iii, instance_name in enumerate(split[dataset][class_name]):
                ds.append(dataset)
                cn.append(class_name)
                inn.append(instance_name)
                exd.append(experiment_directory)
                ckp.append(checkpoint)
                dtd.append(data_dir)
                tot.append(test_or_train)
                cl.append(correspondence_level)
                cpn.append(correspondence_pts_num)
                # results += [evaluate_one_instance(dataset, class_name, instance_name, experiment_directory,
                #                                   checkpoint, data_dir, test_or_train, 
                #                                   corrspondence_level, correspondence_pts_num)]

    print('multi thread start')
    results = p.map(evaluate_one_instance, ds, cn, inn, exd, ckp, dtd, tot, cl, cpn)
    # print('results_length:', len(results))
    # print('q1', results[0])
    # print('q1 length:', len(results[0]))
    # print('q1', results[0])

    chamfer_dist_mean = np.mean([q[1]['chamfer_distance'] for q in results])
    chamfer_dist_median = np.median([q[1]['chamfer_distance'] for q in results])
    earth_mover_dist_mean = np.mean([q[1]['earthmover_distance'] for q in results])
    earth_mover_dist_median = np.median([q[1]['earthmover_distance'] for q in results])
    NMV_ratio_mean = np.mean([q[1]['NM-V'] for q in results])
    NMV_ratio_median = np.median([q[1]['NM-V'] for q in results])
    NME_ratio_mean = np.mean([q[1]['NM-E'] for q in results])
    NME_ratio_median = np.median([q[1]['NM-E'] for q in results])
    NMF_ratio_mean = np.mean([q[1]['NM-F'] for q in results])
    NMF_ratio_median = np.median([q[1]['NM-F'] for q in results])
    self_intersection_ratio_mean = np.mean([q[1]['self-intersection'] for q in results])
    self_intersection_ratio_median = np.median([q[1]['self-intersection'] for q in results])
    normal_consistency_mean = np.mean([q[1]['normal_consistency'] for q in results])
    normal_consistency_median = np.median([q[1]['normal_consistency'] for q in results])
    abs_normal_consistency_mean = np.mean([q[1]['abs_normal_consistency'] for q in results])
    abs_normal_consistency_median = np.median([q[1]['abs_normal_consistency'] for q in results])
    print(chamfer_dist_mean, chamfer_dist_median)
    print(earth_mover_dist_mean, earth_mover_dist_median)
    print(NMV_ratio_mean, NMV_ratio_median)
    print(NME_ratio_mean, NME_ratio_median)
    print(NMF_ratio_mean, NMF_ratio_median)
    print(self_intersection_ratio_mean, self_intersection_ratio_median)
    print(normal_consistency_mean, normal_consistency_median)
    print(abs_normal_consistency_mean, abs_normal_consistency_median)

    suffix = f'_{test_or_train}'
    if correspondence_level is not None:
        cl_suffix = correspondence_level
        cnp_suffix = correspondence_pts_num
        suffix += f'_{cl_suffix}_{cnp_suffix}'

    with open(
        os.path.join(
            ws.get_evaluation_dir(experiment_directory, checkpoint, True), f"chamfer_and_emd_and_nonmanifold{suffix}.csv"
        ),
        "w",
    ) as f:
        f.write("shape, chamfer_dist, earthmovers_dist, NMV_ratio, NME_ratio, NMF_ratio," +\
            " self_intersection_ratio, normal_consistency, abs_normal_consistency\n")
        for result in results:
            f.write(f"{result[0]}, {result[1]['chamfer_distance']}, {result[1]['earthmover_distance']}, " +\
                f"{result[1]['NM-V']}, {result[1]['NM-E']}, {result[1]['NM-F']}, {result[1]['self-intersection']}, " +\
                    f"{result[1]['normal_consistency']}, {result[1]['abs_normal_consistency']}\n")

        f.write(f"CD_Mean, CD_Median, EMD_Mean, EMD_Median, NC_Mean, NC_Median, ANC_Mean, ANC_Median\n")
        f.write(f"{chamfer_dist_mean}, {chamfer_dist_median}, {earth_mover_dist_mean}, {earth_mover_dist_median}, " +\
            f"{normal_consistency_mean}, {normal_consistency_median}, {abs_normal_consistency_mean}, {abs_normal_consistency_median}\n")
        
        f.write(f"NMV_Mean, NMV_Median, NME_Mean, NME_Median, NMF_Mean, NMF_Median, Self_Intersection_Mean, Self_Intersection_Median\n")
        f.write(f"{NMV_ratio_mean}, {NMV_ratio_median}, {NME_ratio_mean}, {NME_ratio_median}, {NMF_ratio_mean}, {NMF_ratio_median}, {self_intersection_ratio_mean}, {self_intersection_ratio_median}\n")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a NDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to evaluate.",
    )
    arg_parser.add_argument(
        "--test_or_train",
        "-t",
        dest="test_or_train",
        required=True,
        help="Whether to evaluate training meshes or reconstructed meshes",
    )
    arg_parser.add_argument(
        "--correspondence_level",
        "-l",
        dest="correspondence_level",
        default=0,
        help="Whether to evaluate meshes generated from template mapping or not," +
        "in which level (coarse or fine)"
    )
    arg_parser.add_argument(
        "--correspondence_pts_num",
        "-n",
        dest="correspondence_pts_num",
        default=0,
        help="if evaluate meshes generated from template mapping, how many vertices in template meshes"
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    if args.correspondence_level == '0':
        args.correspondence_level = None

    evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.split_filename,
        args.test_or_train,
        args.correspondence_level,
        args.correspondence_pts_num
    )
