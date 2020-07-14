import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


##############################
#    Basic layers
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act_type:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


class MLP(Seq):
    def __init__(self, channels, act_type='relu', norm_type=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act_type:
                m.append(act_layer(act_type))
            if norm_type:
                m.append(norm_layer(norm_type, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act_type='relu', norm_type=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act_type:
                m.append(act_layer(act_type))
            if norm_type:
                m.append(norm_layer(norm_type, channels[-1]))
        super(BasicConv, self).__init__(*m)


def batched_index_select(inputs, index):
    """

    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices, _ = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.view(batch_size, -1)

    inputs = inputs.transpose(2, 1).contiguous().view(-1, num_dims)
    index = index.view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.view(-1)

    return torch.index_select(inputs, 0, index).view(batch_size, -1, num_dims).transpose(2, 1).view(batch_size, num_dims, -1, k)

