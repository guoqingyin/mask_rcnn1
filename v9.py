import numpy as np
import os
import torch
import kornia.geometry as kg
import open3d as o3d
import predict
import plyfiletest

def read_K_pose(path_dir):
    cameras = []
    K = np.loadtxt(path_dir+'intrinsics_color.txt')
    K = torch.tensor(K)[None]

    dir_path = os.path.abspath(path_dir+'pose')
    for file_name in sorted(os.listdir(dir_path), key=lambda x: int(x.split('.')[0])):
        pose_path = os.path.join(dir_path, file_name)
        pose = np.loadtxt(os.path.abspath(pose_path))
        pose = np.linalg.inv(pose)
        cameras.append(torch.tensor(pose)[None])
    return K,cameras

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    points_xyz = torch.tensor(pcd.points)
    colors_RGB = (255 * torch.tensor(pcd.colors)).int()
    return points_xyz,colors_RGB


def projection(pose,K,points):
    pinhole_camera = kg.PinholeCamera(torch.tensor(K),torch.tensor(pose),torch.tensor(480)[None], torch.tensor(640)[None])
    intrinsics=pinhole_camera.intrinsics
    extrinsics=pinhole_camera.extrinsics
    P = pinhole_camera.intrinsics @ pinhole_camera.extrinsics
    pts = torch.tensor(points)
    pts_in_camera_coord=kg.transform_points(extrinsics,pts)
    pts_3d_transformed = kg.transform_points(P, pts)
    pts_proj = kg.convert_points_from_homogeneous(pts_3d_transformed)
    return pts_proj[0],pts_3d_transformed[0]

def create_filters(pts_proj,pts_3d_transformed,num_obj,masks):
    height = 968
    width = 1296
    x_filter = (0 < pts_proj[:, 0]) & (pts_proj[:, 0] < width)
    y_filter = (0 < pts_proj[:, 1]) & (pts_proj[:, 1] < height)
    z_filter = pts_3d_transformed[:, 2] > 0
    xyz_filter = x_filter & y_filter & z_filter
    mask_filter = torch.tensor(np.zeros((num_obj, len(x_filter))))
    mask_filter = (mask_filter > 1)

    for obj in range(num_obj):
        for i in range(len(x_filter)):
            if xyz_filter[i] == True:
                x_value = pts_proj[i][0].int()
                y_value = pts_proj[i][1].int()
                if masks[obj][y_value][x_value] == True:
                    mask_filter[obj][i] = True
    total_filter = mask_filter & xyz_filter
    return total_filter
def intersection(path1,path2):
    points1,colors1=load_pcd(path1)
    points2, colors2 = load_pcd(path2)
    pcd1=torch.cat((points1, colors1), 1)
    pcd2 = torch.cat((points2, colors2), 1)
    pcd1_sorted=sorted(pcd1, key = lambda x: (x[0],x[1]))
    pcd2_sorted = sorted(pcd2, key=lambda x: (x[0], x[1]))
    i,j=0,0
    num=0
    list=[]
    while i<len(pcd1_sorted) and j<len(pcd2_sorted):
        x1=np.asarray(pcd2_sorted[i])
        x2=np.asarray(pcd1_sorted[j])
        if (x1==x2).all():
            i+=1
            j+=1
            num+=1
            list.append(x1)
        elif x1[0]>x2[0]:
            j+=1
        else:
            i+=1
    points_colors=torch.tensor(list)
    return points_colors #in tensors
# for posei in range(num):
def main():
    folderpath_father = 'D:\IMAGINE\scene0000_00\\'
    folderpath = folderpath_father + 'stream\\'
    plyname = "scene0000_00_vh_clean_2.ply"
    path_save = folderpath + 'contrast'
    path_color_img = folderpath + 'color'
    # bike1=folderpath_father+"PLY\\"+'002500_0_bicycle_1.0.ply'
    # bike2=folderpath_father+"PLY\\"+'002100_0_bicycle_1.0.ply'
    # points_colors=intersection(bike1,bike2)
    # print('x')
    # list = []
    # for point_color in np.asarray(points_colors):
    #    tuple=(point_color[0], point_color[1], point_color[2], point_color[3], point_color[4], point_color[5])
    #    list.append(tuple)
    # plyfile_name=folderpath_father+'testbike'+'.ply'
    # plyfiletest.write_ply(plyfile_name, list)


    #                 for point_color in np.asarray(points_colors):
    #                    tuple=(point_color[0], point_color[1], point_color[2], point_color[3], point_color[4], point_color[5])
    #                    list.append(tuple)
    #                 plyfile_name=folderpath_father+"PLY\\"+color_img_filename[posei].split(".")[0]+'_'+str(mask0_num)+'_'+category_index[str(classes[mask0_num])]+'_'+str(round(scores[mask0_num],2))+'.ply'
    color_img_filename = sorted(os.listdir(path_color_img), key=lambda x: int(x.split('.')[0]))

    K, cameras_pose = read_K_pose(folderpath)
    points, colors = load_pcd(folderpath_father + plyname)
    num = len(cameras_pose)

    for posei in range(num):
        path_predict=os.path.join(path_color_img,color_img_filename[posei])

        if predict.predict_mask(path_predict)!=None:
            masks,classes,scores,category_index,plot_img=predict.predict_mask(path_predict)
            num_obj=classes.shape[0]
            plot_img.save(folderpath_father+"PLY\\"+color_img_filename[posei].split(".")[0]+".jpg")
            pts_proj, pts_3d_transformed = projection(cameras_pose[posei], K, points)
            total_filter=create_filters(pts_proj,pts_3d_transformed,num_obj,masks)

            for mask0_num in range(num_obj):
                if scores[mask0_num]>=0.7:
                    mask0=total_filter[mask0_num]
                    scene_points_xyz = (pts_proj[mask0]).int()
                    scene_points_color = (colors[mask0]).int()

                    points_seperate=points[0][mask0]
                    colors_seperate=colors[mask0]
                    points_colors=torch.cat((points_seperate,colors_seperate),1)
                    list=[]
                    for point_color in np.asarray(points_colors):
                       tuple=(point_color[0], point_color[1], point_color[2], point_color[3], point_color[4], point_color[5])
                       list.append(tuple)
                    plyfile_name=folderpath_father+"PLY\\"+color_img_filename[posei].split(".")[0]+'_'+str(mask0_num)+'_'+category_index[str(classes[mask0_num])]+'_'+str(round(scores[mask0_num],2))+'.ply'

                    plyfiletest.write_ply(plyfile_name,list)
if __name__=="__main__":
    main()

# img = np.zeros((height, width, 3), np.uint8)
# img[:, :, :] = [255, 255, 255]
# img_distance=np.zeros((height, width))
# flag= 0
# patch = 1
# for point in scene_points_xyz:
#     mycolor = scene_points_color[flag]
#     # no color in this pixel or this pixel already has a color,but there is anthor better color for it
#
#     img[point[1], point[0],:] = mycolor
#     for y in range(point[1] - patch, point[1] + patch +1):
#         for x in range(point[0] - patch, point[0] + patch +1):
#             if x >= 0 and x  < width and y  >= 0 and y  < height:
#                 img[y, x,:] = mycolor
#
#     flag += 1
#
# img_color=cv.imread(os.path.join(path_color_img,color_img_filename[posei]))
# img_compare2=np.hstack([img_color, img])
# #cv.imshow(name,img)
# path_picture=path_save+'\\'+color_img_filename[posei]
# cv.imwrite(path_picture,img_compare2)
# #cv.waitKey()

