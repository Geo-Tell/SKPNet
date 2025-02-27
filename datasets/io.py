import h5py
import numpy as np
import open3d
import os
from plyfile import PlyData, PlyElement

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    # plydata = PlyData.read(filename)
    # pc = plydata['vertex'].data
    # pc_array = np.array([[x, y, z] for x,y,z in pc])
    # return pc_array
    plydata = PlyData.read(filename)
    vert_x = plydata['vertex']['x']
    vert_y = plydata['vertex']['y']
    vert_z = plydata['vertex']['z']
    point_cloud = np.vstack([vert_x, vert_y, vert_z]).T
    return point_cloud


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)



class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]