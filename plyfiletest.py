import numpy as np
from plyfile import PlyData, PlyElement

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z,r,g,b,a]
                         for x,y,z,r,g,b,a in pc],dtype=np.float64)
    return pc_array

def write_ply(output_file,points_color,text=True):
    # points=np.array(points_color[:,0:3],dtype='f8')
    # color_data=np.array()
    points=np.array(points_color,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red','u1'),('green','u1'),('blue','u1')])
    el = PlyElement.describe(points, 'vertex')
    PlyData([el], text=text).write(output_file)
# plydata=PlyData.read("D:\\IMAGINE\\scene0000_00\\scene0000_00_vh_clean_2.ply")
# points_color =plydata['vertex'].data
# write_ply("D:\\IMAGINE\\scene0000_00\\test1.ply",points_color)