import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)

def anisotropic_R_error(r1, r2, seq='xyz', degrees=True):
    '''
    Calculate mse, mae euler agnle error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2)**2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)
    return r_mse, r_mae


def anisotropic_t_error(t1, t2):
    '''
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return t_mse, t_mae


def isotropic_R_error(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.permute(0, 2, 1).contiguous()
    r1r2 = torch.matmul(r2_inv, r1)
    # device = r1.device
    # B = r1.shape[0]
    # mask = torch.unsqueeze(torch.eye(3).to(device), dim=0).repeat(B, 1, 1)
    # tr = torch.sum(torch.reshape(mask * r1r2, (B, 9)), dim=-1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def isotropic_t_error(t1, t2, R2):
    '''
    Calculate isotropic translation error between t1 and t2.
    :param t1: shape=(B, 3), pred_t
    :param t2: shape=(B, 3), gtt
    :param R2: shape=(B, 3, 3), gtR
    :return:
    '''
    R2, t2 = inv_R_t(R2, t2)
    error = torch.squeeze(R2 @ t1[..., None], -1) + t2
    error = torch.norm(error, dim=-1)
    return error




#def modified_CD(tranformed_src, ref):
#    pass


# def rotation_error(r1, r2):
#     '''
#     calculate mse r1-r2 error.
#     :param r1: shape=(B, 3, 3), pred
#     :param r2: shape=(B, 3, 3), gt
#     :return:
#     '''
#     r = torch.reshape(r1 - r2, (-1, 9))
#     error = torch.mean(torch.sum(r ** 2, dim=1))
#     return error


# usage

#  def compute_metrics(self, R, t, igt):
    #  """
    #  R,      [B, 3, 3]   predicted R
    #  t,      [B, 3]      predicted t
    #  igt,    [B, 4, 4]   gt
    #  """
    #  gtR = igt[:, :3, :3]
    #  gtt = igt[:, :3, 3]
    #  inv_R, inv_t = metrics.inv_R_t(gtR, gtt)
    #  cur_r_mse, cur_r_mae = metrics.anisotropic_R_error(R, inv_R)
    #  cur_t_mse, cur_t_mae = metrics.anisotropic_t_error(t, inv_t)
    #  cur_r_isotropic = metrics.isotropic_R_error(R, inv_R)
    #  cur_t_isotropic = metrics.isotropic_t_error(t, inv_t, inv_R)
    #  return cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropic

#  s = self.compute_metrics(R, t, igt)
#  scores = []
#  for i in s:
    #  scores.append(torch.mean(torch.tensor(i)))
#  self.log('val/r_mse', scores[0])
#  self.log('val/r_mae', scores[1])
#  self.log('val/t_mse', scores[2])
#  self.log('val/t_mae', scores[3])
#  self.log('val/r_isotropic', scores[4])
#  self.log('val/t_isotropic', scores[5])
