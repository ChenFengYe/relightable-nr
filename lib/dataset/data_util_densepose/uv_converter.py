import os
import numpy as np
import scipy.io
import pickle

# transform densepose IUV to our uv_map
def TransferDenseposeUV(IUV):
    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    uv_shape = IUV.shape
    uv_map = np.zeros((uv_shape[0], uv_shape[1], 2), dtype=np.float64)
    for i in range(4):
        for j in range(6):
            PartInd = 6 * i + j + 1
            x, y = np.where(IUV[:, :, 0] == PartInd)
            uv_map[x, y, 0] = 1. / 4. * i + U[x, y] / 4.  # u for y
            uv_map[x, y, 1] = 1. / 6. * j + (1 - V[x, y]) / 6.  # v for x, flip each sub v-axis
            uv_map[x, y, 1] = 1 - uv_map[x, y, 1]  # flip back whole v-axis
    uv_map[uv_map >= 1.0] = 1 - 1e-5
    uv_map[uv_map < 0] = 0.

    # transfer densepose uv to SMPL uv map
    return uv_map

class UVConverter(object):
    def __init__(self, relation_path):
        # load uv relation mapping between two uv spaces(densepose and smpl)
        if not os.path.isfile(relation_path):
            raise ValueError('Not exist ' + relation_path)

        self.densepose_to_SMPL = scipy.io.loadmat(relation_path)['densepose_to_SMPL']

    def __call__(self, IUV):
        densepose_size = self.densepose_to_SMPL.shape[1] - 1
        
        # get sub_array (index only)
        ids = IUV[:,:,0]
        uv = IUV[:,:,1:3]
        
        # clean data
        mask = ids!=0
        uv[~mask,:] = 0
        uv[uv>=1.0] = 1 - 1e-5
        uv[uv<0.0] = 0.
          
        id_ = ids[mask].astype(int)
        u_ = (uv[...,0][mask] * densepose_size).astype(int)
        v_ = ((1.0-uv[...,1][mask]) * densepose_size).astype(int)

        uv[mask,:] = self.densepose_to_SMPL[id_-1, u_, v_, 0:2]
        uv_map = IUV[..., 1:3].astype(np.float64)

        # change axis
        uv_map = uv_map[:,:,::-1]
        uv_map[mask, 1] = 1.0 - uv_map[mask, 1]
            
        return uv_map
