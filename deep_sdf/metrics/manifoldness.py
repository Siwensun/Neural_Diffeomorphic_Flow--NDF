import numpy as np
import torch
import open3d as o3d
from mesh_intersection.bvh_search_tree import BVH

def calculate_manifoldness(gen_mesh):
    '''
    This function evaluates a mesh based on the following metrics:
    manifold-edge (using open3D)
    manifold-vertex (using open3D)
    manifold-face  (adjacent face normal calculation). This is also used to compute normal consistecy sometimes
    mesh-intersections (using the torch-mesh-isect library)
    
    gen_mesh: generated triangle mesh
    nv: number of vertices
    ne: number of edges
    nf: number of faces
    nm_edges: number of instances of non-manifold edges
    nm_vertices: number of instances of non-manifold vertices
    nm_faces: number of instances of non-manifold faces
    mesh_isect: number of instances of self-intersections (only 1 out of the two triangles is counted)
    '''
    
    nm_vertices, nm_edges = calculate_non_manifold_edge_vertex(gen_mesh)
    nv, ne, nf, nm_faces, mesh_isect = calculate_non_manifold_face_intersection(gen_mesh)
    
    return nm_vertices/nv, nm_edges/ne, nm_faces/nf, mesh_isect/nf


def calculate_non_manifold_edge_vertex(gen_mesh):
    '''
    This function returns the scores for non-manifold edge and vertices
    gen_mesh: generated triangle mesh
    nm_edges: number of instances of non-manifold edges
    nm_vertices: number of instances of non-manifold vertices
    '''
    gen_v = np.asarray(gen_mesh.vertices)
    gen_f = np.asarray(gen_mesh.faces)

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(gen_v)
    mesh.triangles = o3d.utility.Vector3iVector(gen_f)

    nm_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    nm_vertices = np.asarray(mesh.get_non_manifold_vertices())
    return nm_vertices.shape[0], nm_edges.shape[0]


def calculate_non_manifold_face_intersection(gen_mesh):
    '''
    This function returns the scores for non-manifold faces and amount of self-intersection
    mesh_path: path to the .obj mesh object
    nv: number of vertices
    ne: number of edges
    nf: number of faces
    nm_faces: number of instances of non-manifold faces
    mesh_isect: number of instances of self-intersections (only 1 out of the two triangles is counted)
    '''
    f_adj = gen_mesh.face_adjacency
    faces = gen_mesh.faces
    fn = gen_mesh.face_normals

    count=0
    for f in range(f_adj.shape[0]):
        if fn[f_adj[f,0]]@fn[f_adj[f,1]] < 0:
            count+=1
        
    vertices = torch.tensor(gen_mesh.vertices,
                            dtype=torch.float32).cuda()
    faces = torch.tensor(gen_mesh.faces.astype(np.int64),
                         dtype=torch.long).cuda()

    triangles = vertices[faces].unsqueeze(dim=0).contiguous()

    m = BVH(max_collisions=8)

    outputs = m(triangles)
    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]
    
    return gen_mesh.vertices.shape[0], gen_mesh.edges.shape[0], gen_mesh.faces.shape[0], count, collisions.shape[0]


