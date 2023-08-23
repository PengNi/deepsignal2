
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import warnings
warnings.simplefilter('ignore')
from utils import constants
use_cuda = torch.cuda.is_available()


def squash(tensor, dim=-1):
    squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class Squash(nn.Module):
    def __init__(self, eps=10e-21, **kwargs):
        super(Squash, self).__init__(**kwargs)
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return (1 - 1 / (torch.exp(n) + self.eps)) * (s / (n + self.eps))

def swish(x):
    """Swish activation

    Swish is self-gated linear activation :math:`x sigma(x)`

    For details see: https://arxiv.org/abs/1710.05941

    Note:
        Original definition has a scaling parameter for the gating value,
        making it a generalisation of the (logistic approximation to) the GELU.
        Evidence presented, e.g. https://arxiv.org/abs/1908.08681 that swish-1
        performs comparable to tuning the parameter.

    """
    return x * torch.sigmoid(x)

def dynamic_routing(x, iterations=3):
    # x = x.unsqueeze(-1)
    N = x.shape[1]  # num_caps
    N1 = x.shape[2]  # in_caps
    B = x.shape[0]
    # feature_dim = x.shape[2]
    # x:batch_size, num_caps, in_caps, out_channels
    b = torch.zeros(B, N, N1, 1).to(x.device)  # batch_size, num_caps, in_caps
    for _ in range(iterations):
        # print('input x\'s batch_size: {}, num_caps: {}, in_caps: {}, out_channels: {}'.format(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        c = F.softmax(b, dim=1)  # Softmax along num_caps
        # batch_size, num_caps,caps_dim
        # print('softmax result\'s batch_size: {}, num_caps: {}, in_caps: {}, softmax_result: {}'.format(c.shape[0], c.shape[1], c.shape[2], c.shape[3]))
        a = c * x
        # print('a\'s batch_size: {}, num_caps: {}, in_caps: {}, out_channels: {}'.format(a.shape[0], a.shape[1], a.shape[2], a.shape[3]))
        s = torch.sum(a, dim=2).squeeze(-1)  # sum across in_caps
        # print('s\'s batch_size: {}, num_caps: {}, out_channels: {}'.format(s.shape[0], s.shape[1], s.shape[2]))
        v = squash(s)  # apply "squashing" non-linearity along out_channels
        # print('v\'s batch_size: {}, num_caps: {}, out_channels: {}'.format(v.shape[0], v.shape[1], v.shape[2]))
        # print('x shape: {}'.format(x.shape))
        y = torch.matmul(x, v.unsqueeze(-1))
        # print('y shape: {}'.format(y.shape))
        # print('b shape: {}'.format(b.shape))
        b = b + y

    return v


class PrimaryCapsuleLayer(nn.Module):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    Properties of each capsule s_n are exatracted using a 1D depthwise convolution.

    ...

    Attributes
    ----------
    kernel_size[w]: int
        depthwise conv kernel dimension
    conv_num: int
        number of primary capsules
    feature_dimension: int
        primary capsules dimension (number of properties)
    conv_stride: int
        depthwise conv strides
    Methods
    -------
    forward(inputs)
        compute the primary capsule layer
    """

    def __init__(
        self,
        conv_in=34,#2
        conv_out=4,
        #feature_dimension=256,#272,  # 21 * 5,
        kernel_size=2,
        conv_num=5,
        #sig_len=21,
    ):
        super().__init__()

        #self.conv_out = feature_dimension // conv_num#(conv_num * base_num)
        self.conv_out = conv_out
        self.conv_num = conv_num
        self.primary_capsule_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conv_in,#输入信号通道，词向量维度
                    out_channels=self.conv_out,
                    kernel_size=kernel_size,#第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
                    dilation=conv_dilation,#卷积核元素之间的间距
                    padding="same",
                )
                for conv_dilation in range(1, conv_num + 1)
            ]
        )

    def forward(self, x):

        # print('input feature shape: {}'.format(x.shape))
        capsules = [conv(x) for conv in self.primary_capsule_layer]
        # capsules_reshaped = [
        #    c.reshape(self.conv_num, self.feature_dimension) for c in capsules
        # ]
        output_tensor = torch.cat(capsules, dim=1)
        #(batch_size,self.conv_num*self.conv_out,conv_in)
        return Squash()(output_tensor)


def test_for_primary_capsule_layer():
    input = torch.rand(1, 2, 105)
    layer = PrimaryCapsuleLayer()
    print(layer(input).shape)


class CapsLayer(nn.Module):
    def __init__(
        self, num_capsules=1, 
        in_caps=20,#10, 
        in_channels=256,#272, 
        out_channels=20,  # in_channels=105
        device=0
    ):
        super(CapsLayer, self).__init__()
        self.device=device
        if use_cuda:
            self.W = nn.Parameter(
                torch.randn(1, num_capsules, in_caps, out_channels, in_channels).cuda(self.device))
        else:
            self.W = nn.Parameter(
                torch.randn(1, num_capsules, in_caps, out_channels, in_channels))
        # print('W shape: {}'.format(self.W.shape))

    def forward(self, x):
        # print('CapsLayer input shape: {}'.format(x.shape))
        x = x[:, None, ..., None]  # x.unsqueeze(4).unsqueeze(1)
        # x = x.unsqueeze(-1)
        # print('W shape: {}'.format(self.W.shape))
        # print('CapsLayer input shape: {}'.format(x.shape))
        # print('CapsLayer input expand shape: {}'.format(x[ :, :, None, :].shape))
        # (batch_size, num_caps, num_route_nodes, out_channels, 1)
        # print('x shape: {}'.format(x.shape))
        u_hat = torch.matmul(self.W, x)  # (x @ self.W).squeeze(2)
        # u=u_hat.squeeze(-1)
        u_hat = u_hat.squeeze(-1)
        # batch_size, num_caps, in_caps, out_channels
        # print('u_hat\'s batch_size: {}, num_caps: {}, in_caps: {}, out_channels: {}'.format(u_hat.shape[0], u_hat.shape[1], u_hat.shape[2], u_hat.shape[3]))
        class_capsules = dynamic_routing(u_hat)
        return class_capsules

class ReservoirNet(nn.Module):
    def __init__(self, inSize, resSize, a):
        super(ReservoirNet, self).__init__()
        self.inSize = inSize
        self.resSize = resSize
        self.a = a
        self.Win = (torch.rand([self.resSize, 1 + self.inSize]) - 0.5) * 2.4
        self.W = (torch.rand(self.resSize, self.resSize) - 0.5)
        self.Win[abs(self.Win) > 0.6] = 0
        self.rhoW = max(abs(torch.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.reg = 1e-12
        self.one = torch.ones([1, 1])

class CapsNet(nn.Module):
    def __init__(self, 
                 primary_conv=5,
                 hidden_size=256, 
                 primary_conv_out=4,
                 primary_kernel_size=3,
                 num_capsules=2, 
                 cap_output_num=6, 
                 vocab_size=16,
                 embedding_size=16, 
                 dropout_rate=0.5, 
                 num_layers=3,
                 hlstm_size=128,
                 device=0):
        super(CapsNet, self).__init__()
        self.device=device
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.hlstm_size=hlstm_size
        self.num_layers=num_layers
        self.sig_len = constants.SIG_LEN
        self.lstm_seq = nn.LSTM(self.sig_len, self.hlstm_size, self.num_layers,
                            dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.lstm_sig = nn.LSTM(self.sig_len, self.hlstm_size, self.num_layers,
                            dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.primary_layer = PrimaryCapsuleLayer(conv_in=constants.KMER_LEN*2,
                                                 conv_out=primary_conv_out,conv_num=primary_conv,
                                                 kernel_size=primary_kernel_size)
        self.caps_layer = CapsLayer(num_capsules=num_capsules,in_caps=primary_conv_out*primary_conv,
                                    in_channels=2*hlstm_size,out_channels=cap_output_num,device=self.device)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(cap_output_num*num_capsules, hidden_size)  #
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size)).to(torch.float32)
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size)).to(torch.float32)
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(self, seq, sig):
        seq_emb = self.embed(seq.long())#bacth_size, 17, 16
        #seq_emb = seq_emb.unsqueeze(-1)#seq_emb.reshape(seq_emb.shape[0], 1, -1)
        #sig = sig.unsqueeze(-1)#sig.reshape(sig.shape[0], 1, -1)
        #print('seq_emb shape: {}'.format(seq_emb.shape))
        #print('sig shape: {}'.format(sig.shape))
        #batch_size=seq_emb.shape[0]
        #print(batch_size)
        seq_emb,_=self.lstm_seq(seq_emb.to(torch.float32),self.init_hidden(seq_emb.size(0), self.num_layers,self.hlstm_size))
        sig,_=self.lstm_sig(sig.to(torch.float32),self.init_hidden(sig.size(0), self.num_layers,self.hlstm_size))
        #batch_size,sig_len,2*self.hlstm_size
        #print('seq_emb shape: {}'.format(seq_emb.shape))
        #print('sig shape: {}'.format(sig.shape))
        # to(torch.float32) solve RuntimeError:expected scalar type Double but found Float
        x = torch.cat((seq_emb, sig), dim=1).to(torch.float32)
        #bach_size,kmer_len*2,self.hlstm_size*num_directions

        #x,_ = self.lstm_comb(x,self.init_hidden(x.size(0), self.num_layers,self.hlstm_size))
        
        # seq = self.primary_layer(seq)
        # seq = self.caps_layer(seq)
        # sig = self.primary_layer(sig)
        # sig = self.caps_layer(sig)
        x = self.primary_layer(x)
        x = self.caps_layer(x)
        x=torch.reshape(x,(x.shape[0],-1))
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # print(x.shape)
        #x = x.squeeze(1)

        return x, self.softmax(x)


def test_for_caps_net():
    input1 = torch.rand(1, 1, 105)
    input2 = torch.rand(1, 1, 105)

    model = CapsNet()
    # print(model(input1,input2).shape)


class CapsuleLoss(nn.Module):
    def __init__(self,device=0):
        super(CapsuleLoss, self).__init__()
        weight_rank = torch.from_numpy(np.array([1, 1.0])).float()
        self.device=device
        if use_cuda:
            weight_rank = weight_rank.cuda(self.device)
        self.loss = nn.CrossEntropyLoss(weight=weight_rank)

    def forward(self, classes, labels):
        # classes = classes.reshape(classes.shape[0], 2)
        # labels = labels.reshape(labels.shape[0], 1)
        # print('classes shape: {}'.format(classes.shape))
        # print('labels shape: {}'.format(labels.shape))
        # left = F.relu(0.9 - classes[0], inplace=True) ** 2
        # print('left shape: {}'.format(left.shape))
        # right = F.relu(classes[1] - 0.1, inplace=True) ** 2
        # print('right shape: {}'.format(right.shape))

        # margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        # margin_loss = margin_loss.sum()
        return self.loss(classes, labels)
