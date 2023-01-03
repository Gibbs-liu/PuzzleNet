import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataset import ConcatDataset
import open3d as o3d

import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh
import se_math.transforms as transforms

#  import open3d as o3d


class MovedCADDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform):
        self.dataset = dataset
        self.rigid_transform = rigid_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        up, down, downb, upb, obj, fpc_idx, rpc_idx = self.dataset[index]
        mup = self.rigid_transform(up)
        igt = self.rigid_transform.igt # igt: up-> mup
        mupb = self.rigid_transform(upb)

        return down, mup, igt, up, downb, upb, obj, fpc_idx, rpc_idx

"""
======================================================
optimized datasets
======================================================

"""
def sphere_split(points, z=None, need=False):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=50)
    #  sphere.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    sphere.translate(np.random.rand(3,1)/3)
    sphere_le = sphere
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere_le)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(sphere)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    if not need:
        return up, down
    return up, down, sphere_le

def cylinder_split(points, z=None, need=False):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.4, height=1, resolution=50)
    cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    #  cylinder.translate(np.random.rand(3,1)/3)
    #  cylinder.translate((0,0,0))
    cylinder_le = cylinder
    cylinder = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_le)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cylinder)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    if not need:
        return up, down
    return up, down, cylinder_le

def cone_split(points, z=None, need=False):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=1, height=2, resolution=50)
    cone.translate((0,0,-1))
    cone.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    cone_le = cone
    cone = o3d.t.geometry.TriangleMesh.from_legacy(cone_le)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cone)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    if not need:
        return up, down
    return up, down, cone_le

def plane_split(points, z=None, need=False):
    ##
    # @brief 用平面去切点云
    # @param points     np.array n*3
    # @param z          是否用z
    # @return 上下两个点云
    normal = np.random.rand(3, 1)
    if z is None:
        z = np.random.rand(1) / 3
    dis = np.dot(points, normal) + z
    bool = np.array([dis >= 0]).squeeze(0).squeeze(1)
    up = points[bool]
    bool = np.array([dis < 0]).squeeze(0).squeeze(1)
    down = points[bool]
    if not need:
        return up, down
    return up, down, normal

class CADDataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    同时给点云以边界信息
    切两刀
    """
    def __init__(self, path, mode='train', split_rate=0.9, 
            config=None, name= 'np_out2_all_11000_train_2.npy', 
            split_twice=False, pc_slice=plane_split):
        super().__init__()
        self.split_twice = split_twice
        self.path = path
        self.all = np.load(os.path.join(self.path, name), allow_pickle=True)
        self.split = pc_slice

        #  print(pc_slice)

        assert split_rate <= 1
        assert split_rate > 0
        split = self.all.shape[0] * split_rate
        split = int(split)
        if mode == 'train':
            self.all = self.all[:split]
        elif mode == 'val':
            self.all = self.all[split:]
        elif mode == 'test':
            self.all = np.load(os.path.join(self.path, name.replace("_trian", "_test")), allow_pickle=True)
            

    def __len__(self):
        return self.all.shape[0]

    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, numpoints, pc_dim = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, numpoints)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    def fps(self, points, npoints):
        if points.shape[0]<npoints:
            return None
        N, D = points.shape
        xyz = points[:, :3]
        centroids = np.zeros((npoints,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0,N)
        for i in range(npoints):
            centroids[i] = farthest
            centroid = xyz[farthest,:]
            dist = np.sum((xyz - centroid)**2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = points[centroids.astype(np.int32)]
        return point

    def getitem_non_random(self, index):
        ##
        # @brief        只考虑分成两个部分的情况
        # @param index
        # @return       上、下两部分，上下两个的边界

        pc = self.all[index]
        up, down, obj = self.split(np.array(pc, dtype=np.float32), need=True)
        #  print(up.shape)
        #  print(down.shape)
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            #  print('sample2')
            #  print(up.shape)
            #  print(down.shape)
            up, down, obj = self.split(np.array(pc, dtype=np.float32), need=True)
        self.up = self.fps(up, 1024)        # facade point cloud
        self.down = self.fps(down, 1024)    # roof point cloud

        self.up = torch.from_numpy(self.up).to(torch.float32)
        self.down = torch.from_numpy(self.down).to(torch.float32)

        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)    # fpc boundary

        #  print(self.up.shape, self.down.shape, self.fpcb.shape, self.rpcb.shape)

        return self.up, self.down, self.fpcb, self.rpcb, obj, fpc_idx, rpc_idx

    def slice(self, pc, z, up, down, times=5):
        # 回退
        time = 0
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(pc, z)
        up, down = self.fps(up, 1024), self.fps(down, 1024)
        self.up = torch.from_numpy(up).to(torch.float32)
        self.down = torch.from_numpy(down).to(torch.float32)
        self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)
        return self.up, self.down, self.fpcb, self.rpcb

    def __getitem__(self, index):
        #  print('nonono')
        if not self.split_twice:
            #  print('hello')
            return self.getitem_non_random(index)
        pc = self.all[index]
        slice_seed = torch.randint(0,3, (1,))
        slice_seed = int(slice_seed)
        up, down = self.split(np.array(pc, dtype=np.float32))

        # 节省数据
        if slice_seed == 1 and up.shape[0] < 3000:
            slice_seed = 2
        if slice_seed ==2 and down.shape[0]<3000:
            slice_seed = 1

        if slice_seed ==0:
            # 中间切一刀就够了
            return self.slice(pc, None, up, down)
        elif slice_seed ==1:
            # 上面切一刀
            time = 0
            #  uppc = up
            uppc, downpc = self.split(up, None)
            while time <= 5 and (uppc.shape[0]<1024 or downpc.shape[0]<1024):
                uppc, downpc = self.split(up, None)
                time += 1
            if time>5 and (uppc.shape[0]<1024 or downpc.shape[0]<1024):
                return self.slice(pc, None, up, down)
            if uppc.shape[0]>=1024 and downpc.shape[0]>=1024:
                uppc, downpc = self.fps(uppc, 1024), self.fps(downpc, 1024)
                self.up = torch.from_numpy(uppc).to(torch.float32)
                self.down = torch.from_numpy(downpc).to(torch.float32)
                self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.6:
                    return self.up, self.down, self.fpcb, self.rpcb
                else:
                    """多返回一些，连着上面的up、down等"""
                    if down.shape[0]<1200:
                        return self.up, self.down, self.fpcb, self.rpcb
                    #  down = self.fps(down, 1024)
                    down = torch.from_numpy(down).to(torch.float32)
                    #  cd1, cd2 = self.chamfer_loss(self.up.unsqueeze(0), down.unsqueeze(0))
                    #  cd = torch.mean(cd1) + torch.mean(cd2)
                    #  cd1, cd2 = self.chamfer_loss(self.down.unsqueeze(0), down.unsqueeze(0))
                    #  cdd = torch.mean(cd1)+ torch.mean(cd2)
                    #  if cd > cdd:
                    d = self.down
                    self.down = torch.cat([down, self.up], dim=0)
                    self.up = d
                    #  else:
                        #  d = self.down
                        #  self.down = torch.cat([down, self.up], dim=0)
                        #  self.up = d
                    self.down, self.up = self.fps(self.down.numpy(), 1024), self.fps(self.up.numpy(), 1024)
                    self.up = torch.from_numpy(uppc).to(torch.float32)
                    self.down = torch.from_numpy(downpc).to(torch.float32)
                    self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb

        elif slice_seed == 2:
            # 下面切一刀
            time = 0
            #  uppc = down
            uppc, downpc = self.split(down, None)
            while time <= 5 and (uppc.shape[0]<1024 or downpc.shape[0]<1024):
                uppc, downpc = self.split(down, None)
                time += 1
            if time>5 and (uppc.shape[0]<1024 or downpc.shape[0]<1024):
                return self.slice(pc, None, up, down)
            if uppc.shape[0]>=1024 and downpc.shape[0]>=1024:
                uppc, downpc = self.fps(uppc, 1024), self.fps(downpc, 1024)
                self.up = torch.from_numpy(uppc).to(torch.float32)
                self.down = torch.from_numpy(downpc).to(torch.float32)
                self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.6:
                    return self.up, self.down, self.fpcb, self.rpcb
                else:
                    """多返回一些，连着上面的up、down等"""
                    if up.shape[0] < 1200:
                        return self.up, self.down, self.fpcb, self.rpcb
                    #  up = self.fps(up, 1024)
                    up = torch.from_numpy(up).to(torch.float32)
                    #  cd1, cd2 = self.chamfer_loss(self.up.unsqueeze(0), up.unsqueeze(0))
                    #  cd = torch.mean(cd1) + torch.mean(cd2)
                    #  cd1, cd2 = self.chamfer_loss(self.down.unsqueeze(0), up.unsqueeze(0))
                    #  cdd = torch.mean(cd1)+ torch.mean(cd2)
                    #  if cd > cdd:
                    self.down = torch.cat([up, self.down], dim=0)
                    #  self.up = self.up
                    #  else:
                        #  d = self.down
                        #  self.down = torch.cat([up, self.up], dim=0)
                        #  self.up = d
                    self.down, self.up = self.fps(self.down.numpy(), 1024), self.fps(self.up.numpy(), 1024)
                    self.up = torch.from_numpy(uppc).to(torch.float32)
                    self.down = torch.from_numpy(downpc).to(torch.float32)
                    self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb
                return self.up, self.down, self.fpcb, self.rpcb


    def get_boundary(self, fpc, de_mrpc):
        cd1, cd2 = self.chamfer_loss(fpc.unsqueeze(0), de_mrpc.unsqueeze(0))
        top32cd1 = torch.topk(-cd1, 128)
        cdxyz1 = de_mrpc[top32cd1[1][0]]
        top32cd2 = torch.topk(-cd2, 128)
        cdxyz2 = fpc[top32cd2[1][0]]
        fpc_idx = torch.zeros(1024)
        fpc_idx[top32cd2[1][0]] = 1
        rpc_idx = torch.zeros(1024)
        rpc_idx[top32cd1[1][0]] = 1
        return cdxyz2, cdxyz1, fpc_idx, rpc_idx


class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', split=0.8,  config=None, file='buildings_f_train1024.npy'):
        self.path = path
        self.fpcs = np.load(os.path.join(self.path,file))
        self.rpcs = np.load(os.path.join(self.path,file.replace('_f_', '_r_')))
        assert split <= 1
        assert split > 0
        split = len(self.fpcs) * split
        split = int(split)
        if mode == 'train':
            self.fpcs = self.fpcs[:split]
            self.rpcs = self.rpcs[:split]
        elif mode == 'val':
            self.fpcs = self.fpcs[split:]
            self.rpcs = self.rpcs[split:]
        elif mode == 'test':
            filename = file.replace('_train', '_test')
            self.fpcs = np.load(os.path.join(self.path, filename))
            self.rpcs = np.load(os.path.join(self.path, filename.replace("_f_", "_r_")))

    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, numpoints, pc_dim = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, numpoints)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    def get_boundary(self, fpc, de_mrpc):
        cd1, cd2 = self.chamfer_loss(fpc.unsqueeze(0), de_mrpc.unsqueeze(0))
        top32cd1 = torch.topk(-cd1, 128)
        cdxyz1 = de_mrpc[top32cd1[1][0]]
        top32cd2 = torch.topk(-cd2, 128)
        cdxyz2 = fpc[top32cd2[1][0]]
        fpc_idx = torch.zeros(1024)
        fpc_idx[top32cd2[1][0]] = 1
        rpc_idx = torch.zeros(1024)
        rpc_idx[top32cd1[1][0]] = 1
        return cdxyz2, cdxyz1, fpc_idx, rpc_idx

    def __len__(self):
        flen = self.fpcs.shape[0]
        rlen = self.rpcs.shape[0]
        assert flen==rlen
        return flen

    def __getitem__(self, index):
        #  print(self.fpcs.shape)
        #  print(self.rpcs.shape)
        self.fpc = torch.from_numpy(self.fpcs[index]).to(torch.float32)
        self.rpc = torch.from_numpy(self.rpcs[index]).to(torch.float32)
        self.fpcb, self.rpcb, self.fpcb_idx, self.rpcb_idx = self.get_boundary(self.fpc, self.rpc)
        #  return self.fpc, self.rpc, self.fpcb, self.rpcb
        return self.rpc, self.fpc, self.fpcb, self.rpcb, torch.randn(128, 3), self.fpcb_idx, self.rpcb_idx
                


def get_dataset(category, random=False, random_slice=False):
    cad_datapath = '/home/code/transReg/data/cad'
    building_datapath = '/home/code/fmr/data/'

    # 对只切一刀的情况，不要用太大的数据，节省fps过程的时间。
    if random_slice:
        name = 'np_out2_all_11000_train_2.npy'
    else:
        name = 'np_out_all_6000_train_2.npy'
    bed_name = 'np_ob_all_10000_train_2.npy'

    if category == 'fr':
        # 建筑点云
        traindataset = BuildingDataset(building_datapath, 'train')
        valdataset = BuildingDataset(building_datapath, 'val')
        testdataset = BuildingDataset(building_datapath, 'test')
        
        #  trainset = ModifiedTransformedDataset(traindataset, transforms.RandomTransformSE3(0.8, random))
        #  valset = ModifiedTransformedDataset(valdataset, transforms.RandomTransformSE3(0.8, random))
        #  testset = ModifiedTransformedDataset(testdataset, transforms.RandomTransformSE3(0.8, random))

        #  return trainset, valset, testset
    elif category == 'cadr':
        # 平面切
        traindataset = CADDataset(cad_datapath, 'train', split_twice=random_slice, pc_slice=plane_split, name=name)
        valdataset = CADDataset(cad_datapath, 'val', split_twice=random_slice, pc_slice=plane_split, name=name)
        testdataset = CADDataset(cad_datapath, 'test', split_twice=random_slice, pc_slice=plane_split, name=name)
    elif category == 'cad_cyl':
        # 柱面切
        #  all = np.load(os.path.join(cad_datapath, name), allow_pickle=True)
        traindataset = CADDataset(cad_datapath, 'train', split_twice=random_slice, pc_slice=cylinder_split,)
        valdataset = CADDataset(cad_datapath, 'val', split_twice=random_slice, pc_slice=cylinder_split, )
        testdataset = CADDataset(cad_datapath, 'test', split_twice=random_slice, pc_slice=cylinder_split,) 
    elif category == 'cad_cone':
        # 锥面切
        traindataset = CADDataset(cad_datapath, 'train', 
                split_twice=random_slice, pc_slice=cone_split)
        valdataset = CADDataset(cad_datapath, 'val', 
                split_twice=random_slice, pc_slice=cone_split)
        testdataset = CADDataset(cad_datapath, 'test', 
                split_twice=random_slice, pc_slice=cone_split)
    elif category == 'cad_sphere':
        # 球面切
        traindataset = CADDataset(cad_datapath, 'train', split_twice=random_slice, pc_slice=sphere_split)
        valdataset = CADDataset(cad_datapath, 'val', split_twice=random_slice, pc_slice=sphere_split)
        testdataset = CADDataset(cad_datapath, 'test', split_twice=random_slice, pc_slice=sphere_split)

    elif category == 'bed_sphere':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=bed_name,
                mode='train', split_twice=random_slice, pc_slice=sphere_split)
        valdataset = CADDataset(cad_datapath, name=bed_name,
                mode='val', split_twice=random_slice, pc_slice=sphere_split)
        testdataset = CADDataset(cad_datapath, name=bed_name,
                mode='test', split_twice=random_slice, pc_slice=sphere_split)
    elif category == 'bed_cone':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=bed_name,
                mode='train', split_twice=random_slice, pc_slice=cone_split)
        valdataset = CADDataset(cad_datapath, name=bed_name,
                mode='val', split_twice=random_slice, pc_slice=cone_split)
        testdataset = CADDataset(cad_datapath, name=bed_name,
                mode='test', split_twice=random_slice, pc_slice=cone_split)
    elif category == 'bed_cyl':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=bed_name,
                mode='train', split_twice=random_slice, pc_slice=cylinder_split)
        valdataset = CADDataset(cad_datapath, name=bed_name,
                mode='val', split_twice=random_slice, pc_slice=cylinder_split)
        testdataset = CADDataset(cad_datapath, name=bed_name,
                mode='test', split_twice=random_slice, pc_slice=cylinder_split)
    elif category == 'bedr':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=bed_name,
                mode='train', split_twice=random_slice, pc_slice=plane_split)
        valdataset = CADDataset(cad_datapath, name=bed_name,
                mode='val', split_twice=random_slice, pc_slice=plane_split)
        testdataset = CADDataset(cad_datapath, name=bed_name,
                mode='test', split_twice=random_slice, pc_slice=plane_split)
    else:
        raise RuntimeError('Unknown dataset')

    trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8, random))
    valset = MovedCADDataset(valdataset, transforms.RandomTransformSE3(0.8, random))
    testset = MovedCADDataset(testdataset, transforms.RandomTransformSE3(0.8, random))

    return trainset, valset, testset


