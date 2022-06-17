import numpy as np
import torch
import torch.nn.functional as F

from pytorch3d.ops import knn_gather, knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

def compute_geometric_metrics_points(gdth_mesh, gen_mesh):
    
    with torch.no_grad():
        gt_vertices = torch.from_numpy(gdth_mesh.vertices).float()
        gt_faces = torch.from_numpy(gdth_mesh.faces)
        gt_mesh = Meshes(verts = [gt_vertices], faces = [gt_faces])  
        gt_points, gt_normals = sample_points_from_meshes(gt_mesh, num_samples=10000, return_normals=True)      
        
        pred_vertices = torch.from_numpy(gen_mesh.vertices).float()
        pred_faces = torch.from_numpy(gen_mesh.faces)
        pred_mesh = Meshes(verts = [pred_vertices], faces = [pred_faces])
        pred_points, pred_normals = sample_points_from_meshes(pred_mesh, num_samples=10000, return_normals=True)

        metrics = _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, eps=1e-8)

    return metrics

def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)

    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["normal_consistency"] = normal_dist
        metrics["abs_normal_consistency"] = abs_normal_dist

    # Move all metrics to CPU
    metrics = {k: v.cpu().numpy()[0] for k, v in metrics.items()}
    return metrics