# -*- encoding: utf-8 -*-
'''
@File    :   SavePly.py
@Time    :   2021/6/27 21:32
@Author  :   liu haoyu
@Version :   1.0
@Contact :   gibbsliuhy@gmail.com
@License :   (C)Copyright 2020-2021, ZhangXP group-NLPR-CASIA
@Desc    :   
'''

# here put the import lib
import open3d as o3d
import numpy as np
import functools
import torch


def SavePLY(dir, pc:o3d.geometry.PointCloud):

    point_count = len(np.asarray(pc.points))

    f = open(dir, "w+")
    head = 'ply\nformat ascii 1.0\n' \
           'element vertex %d\n' \
           'property float x\n' \
           'property float y\n' \
           'property float z\n' \
           'property uchar red\n'\
           'property uchar green\n'\
           'property uchar blue\n'\
           'end_header\n' % (point_count)
# 'property uchar red\n'\
# 'property uchar green\n'\
# 'property uchar blue\n'\
    f.write(head)


    # point cloud geometry array
    pcgear = np.asarray(pc.points, dtype=np.float64)        # point cloud color array
    # pcclar = np.asarray(pc.colors, dtype=np.float64)
    for i in range(pcgear.shape[0]):
        line = ""
        line += str(pcgear[i,0])+str(" ")+\
                str(pcgear[i,1])+str(" ")+\
                str(pcgear[i,2])+str(" ")+\
                str("128 128 128")+'\n'
        f.write(line)
    f.close()

def SavePTSnumpy(dir, pc):
    if type(pc) == type(torch.rand(1,1)):
        pc = pc.numpy()
    with open(dir, 'w+') as fout:
        for c in pc:
            fout.write('{0} {1} {2} \n'.format(c[0], c[1], c[2]))
    fout.close()
