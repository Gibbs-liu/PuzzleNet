"""
This file is used to load 3D point cloud for network training
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
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

import json
import open3d as o3d




class ModifiedTransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform):
        self.dataset = dataset
        self.rigid_transform = rigid_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fpc, rpc= self.dataset[index]
        p1 = self.rigid_transform(rpc)
        igt = self.rigid_transform.igt

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return fpc, p1, igt, rpc


class FRDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, split=0.8,  config=None, file='buildings_f_train1024.npy'):
        self.path = path
        self.fpcs = np.load(os.path.join(self.path,file))
        self.rpcs = np.load(os.path.join(self.path,file.replace('_f_', '_r_')))
        assert split <= 1
        assert split > 0
        if split < 1:
            split = len(self.fpcs) * split
            split = int(split)
            if train:
                self.fpcs = self.fpcs[:split]
                self.rpcs = self.rpcs[:split]
            elif not train:
                self.fpcs = self.fpcs[split:]
                self.rpcs = self.rpcs[split:]

    def __len__(self):
        flen = self.fpcs.shape[0]
        rlen = self.rpcs.shape[0]
        assert flen==rlen
        return flen

    def __getitem__(self, index):
        #  print(self.fpcs.shape)
        #  print(self.rpcs.shape)
        return torch.from_numpy(self.fpcs[index]).to(torch.float32),  \
                torch.from_numpy(self.rpcs[index]).to(torch.float32)


# global dataset function, could call to get dataset
def get_datasets():
    # set path and category file for training
    # DONE: 
    #  dataset_path = '/home/code/GPointNet/data'
    #  dataset_path = '/home/data/fr'
    dataset_path = '/home/code/fmr/data/'
    cinfo = None

    # testdataset 就是验证集，这里不考虑太详细的划分方式
    traindataset = FRDataset(dataset_path)
    testdataset = FRDataset(dataset_path, 0)
    
    trainset = ModifiedTransformedDataset(traindataset, transforms.RandomTransformSE3(0.8))
    testset = ModifiedTransformedDataset(testdataset, transforms.RandomTransformSE3(0.8))
    return trainset, testset

class MovedCADDataset2(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, need=False):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.need = need

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        up, down, downb, upb, fpc_idx, rpc_idx = self.dataset[index][:6]
        mup = self.rigid_transform(up)
        igt = self.rigid_transform.igt # igt: up-> mup
        mupb = self.rigid_transform(upb)

        if self.need:
            return down, mup, igt, up, downb, upb, fpc_idx, rpc_idx, self.dataset[index][-1]
        return down, mup, igt, up, downb, upb, fpc_idx, rpc_idx
        

class MovedCADDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform):
        self.dataset = dataset
        self.rigid_transform = rigid_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        up, down = self.dataset[index]
        mup = self.rigid_transform(up)
        igt = self.rigid_transform.igt # igt: up-> mup

        return down, mup, igt, up


class bs_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, split=0.75, config=None):
        self.train = train
        self.path = '/home/code/transReg/data/resso/'
        self.up = np.load(os.path.join(self.path, 'bs_up.npy'))
        self.down = np.load(os.path.join(self.path, 'bs_down.npy'))
        self.igt = np.load(os.path.join(self.path, 'bs_igt.npy'))
        self.mup = np.load(os.path.join(self.path, 'bs_mup.npy'))
        assert split < 1
        assert split > 0
        assert self.up.shape == self.down.shape
        split = self.up.shape[0] * split
        split = int(split)
        self.split = split
        if train:
            self.up = self.up[:split]
            self.down = self.down[:split]
        elif not train:
            self.up = self.up[split:]
            self.down = self.down[split:]

    def __len__(self):
        uplen = self.up.shape[0]
        downlen = self.down.shape[0]
        assert uplen == downlen
        return uplen

    def __getitem__(self, index):
        if self.train:
            return torch.from_numpy(self.up[index]).to(torch.float32), \
                    torch.from_numpy(self.down[index]).to(torch.float32), \
                    torch.from_numpy(self.igt[index]).to(torch.float32), \
                    torch.from_numpy(self.mup[index]).to(torch.float32), \
                    torch.tensor(index)
        elif not self.train:
            return torch.from_numpy(self.up[index]).to(torch.float32), \
                    torch.from_numpy(self.down[index]).to(torch.float32), \
                    torch.from_numpy(self.igt[index]).to(torch.float32), \
                    torch.from_numpy(self.mup[index]).to(torch.float32), \
                    torch.tensor(index + self.split)


class snp_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, split=0.75, config=None):
        self.train = train
        self.path = '/home/code/transReg/data/shapenet_part/'
        self.up = np.load(os.path.join(self.path, 'spn_airplane_up.npy'))
        self.down = np.load(os.path.join(self.path, 'spn_airplane_down.npy'))
        self.igt = np.load(os.path.join(self.path, 'spn_airplane_igt.npy'))
        self.mup = np.load(os.path.join(self.path, 'spn_airplane_mup.npy'))
        assert split < 1
        assert split > 0
        assert self.up.shape == self.down.shape
        split = self.up.shape[0] * split
        split = int(split)
        self.split = split
        if train:
            self.up = self.up[:split]
            self.down = self.down[:split]
        elif not train:
            self.up = self.up[split:]
            self.down = self.down[split:]

    def __len__(self):
        uplen = self.up.shape[0]
        downlen = self.down.shape[0]
        assert uplen == downlen
        return uplen

    def __getitem__(self, index):
        if self.train:
            return torch.from_numpy(self.up[index]).to(torch.float32), \
                    torch.from_numpy(self.down[index]).to(torch.float32), \
                    torch.from_numpy(self.igt[index]).to(torch.float32), \
                    torch.from_numpy(self.mup[index]).to(torch.float32), \
                    torch.tensor(index)
        elif not self.train:
            return torch.from_numpy(self.up[index]).to(torch.float32), \
                    torch.from_numpy(self.down[index]).to(torch.float32), \
                    torch.from_numpy(self.igt[index]).to(torch.float32), \
                    torch.from_numpy(self.mup[index]).to(torch.float32), \
                    torch.tensor(index + self.split)





class cad_dataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, split=0.8, config=None, name= 'np_oa_up_train.npy'):
        self.path = path
        self.up = np.load(os.path.join(self.path, name))
        self.down = np.load(os.path.join(self.path, name.replace('_up_', '_down_')))
        assert split <= 1
        assert split > 0
        #  assert self.up.shape == self.down.shape
        if split < 1:
            split = self.up.shape[0] * split
            split = int(split)
            if train:
                self.up = self.up[:split]
                self.down = self.down[:split]
            elif not train:
                self.up = self.up[split:]
                self.down = self.down[split:]

    def __len__(self):
        uplen = self.up.shape[0]
        downlen = self.down.shape[0]
        assert uplen == downlen
        return uplen

    def __getitem__(self, index):
        return torch.from_numpy(self.up[index]).to(torch.float32), \
                torch.from_numpy(self.down[index]).to(torch.float32)

class cad_3_dataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    同时给点云以边界信息
    切两刀
    """
    def __init__(self, path, train=True, split=0.9, config=None, name= 'np_out2_all_11000_train_2.npy', random_add=False) -> None:
        super().__init__()
        self.random_add = random_add
        self.path = path
        self.all = np.load(os.path.join(self.path, name), allow_pickle=True)
        assert split <= 1
        assert split > 0
        if split < 1:
            split = self.all.shape[0] * split
            split = int(split)
            if train:
                self.all = self.all[:split]
            elif not train:
                self.all = self.all[split:]

    def __len__(self):
        return self.all.shape[0]

    def split(self, points, z=None):
        normal = np.random.rand(3, 1)
        if z is None:
            z = np.random.rand(1) / 3
        dis = np.dot(points, normal) + z
        bool = np.array([dis >= 0]).squeeze(0).squeeze(1)
        up = points[bool]
        bool = np.array([dis < 0]).squeeze(0).squeeze(1)
        down = points[bool]
        return up, down

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
        pc = self.all[index]
        up, down = self.split(np.array(pc, dtype=np.float32))
        if self.random_add:
            ju = (torch.rand(1) >=0.4)
            if ju:
                while up.shape[0] < 1024 or down.shape[0] < 1024:
                    up, down = self.split(np.array(pc, dtype=np.float32))
                up, down = self.fps(up, 1024), self.fps(down, 1024)
                self.fpcb, self.rpcb = self.get_boundary(up, down)
                return up, down, self.fpcb, self.rpcb
        while max(up.shape[0], down.shape[0]) < 3500:
            #  print('sample')
            up, down = self.split(np.array(pc, dtype=np.float32))
        if up.shape[0] > down.shape[0]:
            pc = up
        else:
            pc = down
        up, down = self.split(np.array(pc, dtype=np.float32))
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            #  print('sample2')
            #  print(up.shape)
            #  print(down.shape)
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)
        self.down = self.fps(down, 1024)

        self.up = torch.from_numpy(self.up).to(torch.float32)
        self.down = torch.from_numpy(self.down).to(torch.float32)

        self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)

        return self.up, self.down, self.fpcb, self.rpcb 

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
        if not self.random_add:
            return self.getitem_non_random(index)
        pc = self.all[index]
        slice_seed = torch.randint(0,3, (1,))
        slice_seed = int(slice_seed)
        up, down = self.split(np.array(pc, dtype=np.float32))
        if slice_seed ==0:
            # 中间切一刀就够了
            return self.slice(pc, None, up, down)
        elif slice_seed ==1:
            # 上面切一刀
            if up.shape[0] < 3000:
                return self.slice(pc, None, up, down)
            time = 0
            uppc = up
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
                return self.up, self.down, self.fpcb, self.rpcb
        elif slice_seed == 2:
            # 下面切一刀
            if down.shape[0] < 3000:
                return self.slice(pc, None, up, down)
            time = 0
            uppc = down
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
                return self.up, self.down, self.fpcb, self.rpcb


    def get_boundary(self, fpc, de_mrpc):
        cd1, cd2 = self.chamfer_loss(fpc.unsqueeze(0), de_mrpc.unsqueeze(0))
        top32cd1 = torch.topk(-cd1, 128)
        cdxyz1 = de_mrpc[top32cd1[1][0]]
        top32cd2 = torch.topk(-cd2, 128)
        cdxyz2 = fpc[top32cd2[1][0]]
        return cdxyz2, cdxyz1



class cad_1_dataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    """
    def __init__(self, path, train=True, split=0.8, config=None, name= 'np_oa_all_train.npy') -> None:
        super().__init__()
        self.path = path
        self.all = np.load(os.path.join(self.path, name), allow_pickle=True)
        assert split <= 1
        assert split > 0
        if split < 1:
            split = self.all.shape[0] * split
            split = int(split)
            if train:
                self.all = self.all[:split]
            elif not train:
                self.all = self.all[split:]

    def __len__(self):
        return self.all.shape[0]

    def split(self, points):
        normal = np.random.rand(3, 1)
        dis = np.dot(points, normal)
        bool = np.array([dis >= 0]).squeeze(0).squeeze(1)
        up = points[bool]
        bool = np.array([dis < 0]).squeeze(0).squeeze(1)
        down = points[bool]
        return up, down

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

    def __getitem__(self, index):
        pc = self.all[index]
        up, down = self.split(np.array(pc, dtype=np.float32))
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)
        self.down = self.fps(down, 1024)
        return torch.from_numpy(self.up).to(torch.float32),\
                torch.from_numpy(self.down).to(torch.float32)



class cad_2_dataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    同时给点云以边界信息
    """
    def __init__(self, path, train=True, split=0.8, config=None, name= 'np_oa_all_train.npy') -> None:
        super().__init__()
        self.path = path
        self.all = np.load(os.path.join(self.path, name), allow_pickle=True)
        assert split <= 1
        assert split > 0
        if split < 1:
            split = self.all.shape[0] * split
            split = int(split)
            if train:
                self.all = self.all[:split]
            elif not train:
                self.all = self.all[split:]

    def __len__(self):
        return self.all.shape[0]

    def split(self, points):
        normal = np.random.rand(3, 1)
        dis = np.dot(points, normal)
        bool = np.array([dis >= 0]).squeeze(0).squeeze(1)
        up = points[bool]
        bool = np.array([dis < 0]).squeeze(0).squeeze(1)
        down = points[bool]
        return up, down

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

    def __getitem__(self, index):
        pc = self.all[index]
        up, down = self.split(np.array(pc, dtype=np.float32))
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)
        self.down = self.fps(down, 1024)

        self.up = torch.from_numpy(self.up).to(torch.float32)
        self.down = torch.from_numpy(self.down).to(torch.float32)

        self.fpcb, self.rpcb = self.get_boundary(self.down, self.up)

        return self.up, self.down, self.fpcb, self.rpcb 

    def get_boundary(self, fpc, de_mrpc):
        cd1, cd2 = self.chamfer_loss(fpc.unsqueeze(0), de_mrpc.unsqueeze(0))
        top32cd1 = torch.topk(-cd1, 128)
        cdxyz1 = de_mrpc[top32cd1[1][0]]
        top32cd2 = torch.topk(-cd2, 128)
        cdxyz2 = fpc[top32cd2[1][0]]
        return cdxyz2, cdxyz1



class cad_1_dataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    """
    def __init__(self, path, train=True, split=0.8, config=None, name= 'np_oa_all_train.npy') -> None:
        super().__init__()
        self.path = path
        self.all = np.load(os.path.join(self.path, name), allow_pickle=True)
        assert split < 1
        assert split > 0
        split = self.all.shape[0] * split
        split = int(split)
        if train:
            self.all = self.all[:split]
        elif not train:
            self.all = self.all[split:]

    def __len__(self):
        return self.all.shape[0]

    def split(self, points):
        normal = np.random.rand(3, 1)
        dis = np.dot(points, normal)
        bool = np.array([dis >= 0]).squeeze(0).squeeze(1)
        up = points[bool]
        bool = np.array([dis < 0]).squeeze(0).squeeze(1)
        down = points[bool]
        return up, down

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

    def __getitem__(self, index):
        pc = self.all[index]
        up, down = self.split(np.array(pc, dtype=np.float32))
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)
        self.down = self.fps(down, 1024)
        return torch.from_numpy(self.up).to(torch.float32),\
                torch.from_numpy(self.down).to(torch.float32)


def get_cad_datasets(category='cad', random=False, random_slice=False):
    if category == 'cad':
        dataset_path = '/home/code/transReg/data/cad'
        traindataset = cad_dataset(dataset_path)
        testdatset = cad_dataset(dataset_path, 0)

        trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8, True))
        testset = MovedCADDataset(testdatset, transforms.RandomTransformSE3(0.8, True))

        return trainset, testset

    elif category == 'cadrr2':
        """
        随机切, 切两刀
        """
        dataset_path = '/home/code/transReg/data/cad'
        #  traindataset = cad_dataset(dataset_path)
        #  testdatset = cad_dataset(dataset_path, 0)
        traindataset = cad_3_dataset(dataset_path, random_add=random_slice)
        testdatset = cad_3_dataset(dataset_path, 0, random_add=random_slice)

        trainset = MovedCADDataset2(traindataset, transforms.RandomTransformSE3(0.8, random))
        testset = MovedCADDataset2(testdatset, transforms.RandomTransformSE3(0.8, random))
        return trainset, testset

    elif category == 'cadrr':
        """
        随机切 + 边界
        """
        dataset_path = '/home/code/transReg/data/cad'
        #  traindataset = cad_dataset(dataset_path)
        #  testdatset = cad_dataset(dataset_path, 0)
        traindataset = cad_2_dataset(dataset_path)
        testdatset = cad_2_dataset(dataset_path, 0)

        trainset = MovedCADDataset2(traindataset, transforms.RandomTransformSE3(0.8, random))
        testset = MovedCADDataset2(testdatset, transforms.RandomTransformSE3(0.8, random))

        return trainset, testset
    elif category == 'cadr':
        """
        随机切
        """
        dataset_path = '/home/code/transReg/data/cad'
        #  traindataset = cad_dataset(dataset_path)
        #  testdatset = cad_dataset(dataset_path, 0)
        traindataset = cad_1_dataset(dataset_path)
        testdatset = cad_1_dataset(dataset_path, 0)

        trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8, random))
        testset = MovedCADDataset(testdatset, transforms.RandomTransformSE3(0.8, random))

        return trainset, testset

    if category == 'cadpro':
        """
        圆柱切
        """
        dataset_path = '/home/code/transReg/data/cad'
        traindataset = cad_dataset(dataset_path, name='np_oa_cylinder_up_train_2.npy')
        testdatset = cad_dataset(dataset_path, 0, name='np_oa_cylinder_up_train_2.npy')

        trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8))
        testset = MovedCADDataset(testdatset, transforms.RandomTransformSE3(0.8))

        return trainset, testset

    elif category == 'bs':
        traindataset = bs_dataset()
        testdatset = bs_dataset(train=False)

        return traindataset, testdatset

    elif category == 'snp':
        traindataset = snp_dataset()
        testdatset = snp_dataset(train=False)
        return traindataset, testdatset
    elif category == 'cadpro_cone':
        dataset_path = '/home/code/transReg/data/cad'
        traindataset = cad_dataset(dataset_path, name='np_oa_cone_up_train_2.npy')
        testdatset = cad_dataset(dataset_path, 0, name='np_oa_cone_up_train_2.npy')

        trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8))
        testset = MovedCADDataset(testdatset, transforms.RandomTransformSE3(0.8))
    elif category == 'cadpro_sphere':
        dataset_path = '/home/code/transReg/data/cad'
        traindataset = cad_dataset(dataset_path, name='np_oa_sphere_up_train_2.npy')
        testdatset = cad_dataset(dataset_path, 0, name='np_oa_sphere_up_train_2.npy')

        trainset = MovedCADDataset(traindataset, transforms.RandomTransformSE3(0.8))
        testset = MovedCADDataset(testdatset, transforms.RandomTransformSE3(0.8))

        return trainset, testset





"""
======================================================
optimized datasets
======================================================

"""
def sphere_split(points, z=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=50)
    #  sphere.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    sphere.translate(np.random.rand(3,1)/3)
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(sphere)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    return up, down

def cylinder_split(points, z=None):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.6, height=1, resolution=50)
    cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    cylinder.translate(np.random.rand(3,1)/3)
    #  cylinder.translate((0,0,0))
    cylinder = o3d.t.geometry.TriangleMesh.from_legacy(cylinder)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cylinder)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    return up, down

def cone_split(points, z=None):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=1, height=2, resolution=50)
    cone.translate((0,0,-1))
    cone.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3,1)), (0,0,0))
    cone = o3d.t.geometry.TriangleMesh.from_legacy(cone)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cone)
    qurey = o3d.core.Tensor([points], dtype=o3d.core.Dtype.Float32)
    res = scene.compute_signed_distance(qurey)
    res = res.numpy().squeeze()
    bools = np.array([res<0]).squeeze()
    up = points[bools]
    down = points[~bools]
    return up, down

def plane_split(points, z=None):
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
    return up, down

def get_test_dataset(category, random=False, random_slice=False):
    if category == 'cadrr':
        dataset_path = '/home/code/transReg/data/cad'
        dataset = cad_2_dataset(dataset_path, split=1, name='np_oa_all_test.npy')
        tstdataset = MovedCADDataset2(dataset, transforms.RandomTransformSE3(0.8, random))
        return tstdataset
    elif category == 'cadrr2':
        dataset_path = '/home/code/transReg/data/cad'
        dataset = cad_3_dataset(dataset_path, split=1, name='np_oa_all_test.npy', random_add=random_slice)
        tstdataset = MovedCADDataset2(dataset, transforms.RandomTransformSE3(0.8, random))
        return tstdataset


class CADDataset2(torch.utils.data.Dataset):
    """
    backup
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
            self.all = np.load(os.path.join(self.path, name.replace("_train", "_test")), allow_pickle=True)
            

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
        up, down = self.split(np.array(pc, dtype=np.float32))
        #  print(up.shape)
        #  print(down.shape)
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            #  print('sample2')
            #  print(up.shape)
            #  print(down.shape)
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)        # facade point cloud
        self.down = self.fps(down, 1024)    # roof point cloud

        self.up = torch.from_numpy(self.up).to(torch.float32)
        self.down = torch.from_numpy(self.down).to(torch.float32)

        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)    # fpc boundary

        #  print(self.up.shape, self.down.shape, self.fpcb.shape, self.rpcb.shape)

        return self.up, self.down, self.fpcb, self.rpcb , fpc_idx, rpc_idx

    def slice(self, pc, z, up, down, times=5):
        # 回退
        time = 0
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(pc, z)
        up, down = self.fps(up, 1024), self.fps(down, 1024)
        self.up = torch.from_numpy(up).to(torch.float32)
        self.down = torch.from_numpy(down).to(torch.float32)
        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx

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
                self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.6:
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
                else:
                    """多返回一些，连着上面的up、down等"""
                    if down.shape[0]<1200:
                        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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
                    self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx

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
                self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.6:
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
                else:
                    """多返回一些，连着上面的up、down等"""
                    if up.shape[0] < 1200:
                        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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
                    self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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


class BreakingDataset(torch.utils.data.Dataset):
    """
    输入的时候每个点云作为一个完整的点云输入，没有分成上下两部分，每次取的时候随机切一刀
    同时给点云以边界信息
    切两刀
    """
    def __init__(self, mode='train',):
        super().__init__()
        json0_file = "./config/tmp.json"
        json1_file = "./config/tiny_only.json"
        json0_file = open(json0_file, 'r')
        json1_file = open(json1_file, 'r')
        self.mapping0 = json.load(json0_file)[mode+"_maps_tiny"]
        self.mapping1 = json.load(json1_file)[mode+"_maps_tiny"]
        self.items_list = self.mapping0+self.mapping1
        
        self.mode = mode

    def __len__(self):
        assert len(self.mapping0)+len(self.mapping1) == len(self.items_list)
        return len(self.mapping0)+len(self.mapping1)

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
    
    def __getitem__(self, index):
        items = self.items_list[index]
        item0 = items[0]
        item1 = items[1]
        if type(item0)==list:
            points1 = []
            for m in item0:
                mesh = o3d.io.read_triangle_mesh(os.path.join('/home/code/transReg/data', m))
                pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, 2000)
                pc = np.array(pc.points)
                points1.append(pc)
            points1 = np.concatenate(points1, 0)
        else:
            mesh = o3d.io.read_triangle_mesh(os.path.join('/home/code/transReg/data', item0))
            pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, 2000)
            points1 = np.array(pc.points)
        mesh2 = o3d.io.read_triangle_mesh(os.path.join('/home/code/transReg/data', item1))
        pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh2, 2000)
        points2 = np.array(pc.points)
        points1 = self.fps(points1, 1024)
        points2 = self.fps(points2, 1024)
        points1 = torch.from_numpy(points1).to(torch.float32)
        points2 = torch.from_numpy(points2).to(torch.float32)
        fpcb, rpcb, fpc_idx, rpc_idx = self.get_boundary(points1, points2)
        if self.mode == 'train':
            return points2, points1, fpcb, rpcb, fpc_idx, rpc_idx
        if self.mode == 'test':
            return points2, points1, fpcb, rpcb, fpc_idx, rpc_idx, items
        


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
            self.all = np.load(os.path.join(self.path, name.replace("_train", "_test")), allow_pickle=True)
            

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
        up, down = self.split(np.array(pc, dtype=np.float32))
        #  print(up.shape)
        #  print(down.shape)
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            #  print('sample2')
            #  print(up.shape)
            #  print(down.shape)
            up, down = self.split(np.array(pc, dtype=np.float32))
        self.up = self.fps(up, 1024)        # facade point cloud
        self.down = self.fps(down, 1024)    # roof point cloud

        self.up = torch.from_numpy(self.up).to(torch.float32)
        self.down = torch.from_numpy(self.down).to(torch.float32)

        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)    # fpc boundary

        #  print(self.up.shape, self.down.shape, self.fpcb.shape, self.rpcb.shape)

        return self.up, self.down, self.fpcb, self.rpcb , fpc_idx, rpc_idx

    def slice(self, pc, z, up, down, times=5):
        # 回退
        time = 0
        while up.shape[0] < 1024 or down.shape[0] < 1024:
            up, down = self.split(pc, z)
        up, down = self.fps(up, 1024), self.fps(down, 1024)
        self.up = torch.from_numpy(up).to(torch.float32)
        self.down = torch.from_numpy(down).to(torch.float32)
        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx

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
                se = int(torch.randint(0, 3, (1,)))
                if se == 0 or down.shape[0]<1024:
                    # up_down and up_up are composed with up
                    choice = int(torch.randint(0,2,(1,)))
                    up_up = [uppc, downpc][choice]
                    up_up = self.fps(up_up, 1024)
                    up_down = self.fps(np.vstack(([uppc,downpc][(choice+1)%2],down)), 1024)
                    up_up = torch.from_numpy(up_up).to(torch.float32)
                    up_down = torch.from_numpy(up_down).to(torch.float32)
                    fpcb, rpcb, fpc_idx, rpc_idx = self.get_boundary(up_down, up_up)
                    return up_up, up_down, fpcb, rpcb, fpc_idx, rpc_idx
                elif se == 1:
                    # up_down and up_up are composed with up
                    choice = int(torch.randint(0,2,(1,)))
                    up_up = [uppc, downpc][choice]
                    up_up = self.fps(up_up, 1024)
                    up_down = self.fps(down, 1024)
                    up_up = torch.from_numpy(up_up).to(torch.float32)
                    up_down = torch.from_numpy(up_down).to(torch.float32)
                    fpcb, rpcb, fpc_idx, rpc_idx = self.get_boundary(up_down, up_up)
                    cd1, cd2 = self.chamfer_loss(fpcb.unsqueeze(0), rpcb.unsqueeze(0))
                    cd = torch.mean(cd1)+torch.mean(cd2)
                    if float(cd) > 0.015:
                        return self.slice(pc, None, up, down)
                    return up_up, up_down, fpcb, rpcb, fpc_idx, rpc_idx
                uppc, downpc = self.fps(uppc, 1024), self.fps(downpc, 1024)
                self.up = torch.from_numpy(uppc).to(torch.float32)
                self.down = torch.from_numpy(downpc).to(torch.float32)
                self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.7:
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
                else:
                    """多返回一些，连着上面的up、down等"""
                    if down.shape[0]<1200:
                        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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
                    self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx

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
                se = int(torch.randint(0, 3, (1,)))
                if se == 0 or up.shape[0]<1024:
                    #down_down and down_up are composed with up
                    choice = int(torch.randint(0,2,(1,)))
                    down_up = [uppc,downpc][choice]
                    down_up = self.fps(down_up, 1024)
                    down_down = self.fps(np.vstack(([uppc, downpc][(choice+1)%2], up)), 1024)
                    down_up = torch.from_numpy(down_up).to(torch.float32)
                    down_down = torch.from_numpy(down_down).to(torch.float32)
                    fpcb, rpcb, fpc_idx, rpc_idx = self.get_boundary(down_down, down_up)
                    return down_up, down_down, fpcb, rpcb, fpc_idx, rpc_idx
                elif se == 1:
                    #down_down and down_up are composed with up
                    choice = int(torch.randint(0,2,(1,)))
                    down_up = [uppc,downpc][choice]
                    down_up = self.fps(down_up, 1024)
                    down_down = self.fps(up, 1024)
                    down_up = torch.from_numpy(down_up).to(torch.float32)
                    down_down = torch.from_numpy(down_down).to(torch.float32)
                    fpcb, rpcb, fpc_idx, rpc_idx = self.get_boundary(down_down, down_up)
                    cd1, cd2 = self.chamfer_loss(fpcb.unsqueeze(0), rpcb.unsqueeze(0))
                    cd = torch.mean(cd1) + torch.mean(cd2)
                    if float(cd) > 0.015:
                        return self.slice(pc, None, up, down)
                    return down_up, down_down, fpcb, rpcb, fpc_idx, rpc_idx
                uppc, downpc = self.fps(uppc, 1024), self.fps(downpc, 1024)
                self.up = torch.from_numpy(uppc).to(torch.float32)
                self.down = torch.from_numpy(downpc).to(torch.float32)
                self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                re_now = np.random.rand(1)
                if re_now > 0.6:
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
                else:
                    """多返回一些，连着上面的up、down等"""
                    if up.shape[0] < 1200:
                        return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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
                    self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.down, self.up)
                    return self.up, self.down, self.fpcb, self.rpcb, fpc_idx, rpc_idx
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
        #  import pdb
        #  pdb.set_trace()
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
        self.fpcb, self.rpcb, fpc_idx, rpc_idx = self.get_boundary(self.fpc, self.rpc)
        #  return self.fpc, self.rpc, self.fpcb, self.rpcb
        return self.rpc, self.fpc, self.fpcb, self.rpcb, fpc_idx, rpc_idx
                



def get_dataset(category, random=False, random_slice=False):
    cad_datapath = '/home/code/transReg/data/cad'
    building_datapath = '/home/code/fmr/data/'

    # 对只切一刀的情况，不要用太大的数据，节省fps过程的时间。
    if random_slice:
        name = 'np_out2_all_11000_train_2.npy'
    else:
        name = 'np_out_all_6000_train_2.npy'
    name = 'np_out2_all_11000_train_2.npy'
    bed_name = 'np_ob_all_10000_train_2.npy'
    vase_name = 'np_vase_all_11000_train_2.npy'

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
    elif category == 'vaser':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=vase_name,
                mode='train', split_twice=random_slice, pc_slice=plane_split)
        valdataset = CADDataset(cad_datapath, name=vase_name,
                mode='val', split_twice=random_slice, pc_slice=plane_split)
        testdataset = CADDataset(cad_datapath, name=vase_name,
                mode='test', split_twice=random_slice, pc_slice=plane_split)
    elif category == 'vase_cone':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=vase_name,
                mode='train', split_twice=random_slice, pc_slice=cone_split)
        valdataset = CADDataset(cad_datapath, name=vase_name,
                mode='val', split_twice=random_slice, pc_slice=cone_split)
        testdataset = CADDataset(cad_datapath, name=vase_name,
                mode='test', split_twice=random_slice, pc_slice=cone_split)
    elif category == 'vase_cyl':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=vase_name,
                mode='train', split_twice=random_slice, pc_slice=cylinder_split)
        valdataset = CADDataset(cad_datapath, name=vase_name,
                mode='val', split_twice=random_slice, pc_slice=cylinder_split)
        testdataset = CADDataset(cad_datapath, name=vase_name,
                mode='test', split_twice=random_slice, pc_slice=cylinder_split)
    elif category == 'vase_sphere':
        # 球面切
        traindataset = CADDataset(cad_datapath, name=vase_name,
                mode='train', split_twice=random_slice, pc_slice=sphere_split)
        valdataset = CADDataset(cad_datapath, name=vase_name,
                mode='val', split_twice=random_slice, pc_slice=sphere_split)
        testdataset = CADDataset(cad_datapath, name=vase_name,
                mode='test', split_twice=random_slice, pc_slice=sphere_split)
    elif category == 'bbv':
        traindataset = BreakingDataset(mode='train')
        valdataset = BreakingDataset(mode='test')
        testdataset = BreakingDataset(mode='test')
    else:
        raise RuntimeError('Unknown dataset')

    trainset = MovedCADDataset2(traindataset, transforms.RandomTransformSE3(0.8, random))
    valset = MovedCADDataset2(valdataset, transforms.RandomTransformSE3(0.8, random))
    testset = MovedCADDataset2(testdataset, transforms.RandomTransformSE3(0.8, random))

    return trainset, valset, testset


