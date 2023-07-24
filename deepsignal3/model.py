
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import warnings
warnings.simplefilter('ignore')

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
        conv_in=2,
        feature_dimension=272,  # 21 * 5,
        kernel_size=2,
        conv_num=5,
        base_num=21,
    ):
        super().__init__()

        self.conv_out = feature_dimension // (conv_num * base_num)
        self.conv_num = conv_num
        self.primary_capsule_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    self.conv_out,
                    kernel_size,
                    dilation=conv_stride,
                    padding="same",
                )
                for conv_stride in range(1, conv_num + 1)
            ]
        )

    def forward(self, x):

        # print('input feature shape: {}'.format(x.shape))
        capsules = [conv(x) for conv in self.primary_capsule_layer]
        # capsules_reshaped = [
        #    c.reshape(self.conv_num, self.feature_dimension) for c in capsules
        # ]
        output_tensor = torch.cat(capsules, dim=1)
        return Squash()(output_tensor)


def test_for_primary_capsule_layer():
    input = torch.rand(1, 2, 105)
    layer = PrimaryCapsuleLayer()
    print(layer(input).shape)


class CapsLayer(nn.Module):
    def __init__(
        self, num_capsules=1, in_caps=10, in_channels=272, out_channels=2  # in_channels=105
    ):
        super(CapsLayer, self).__init__()
        self.W = nn.Parameter(
            torch.randn(1, num_capsules, in_caps, out_channels, in_channels)
        )
        # print('W shape: {}'.format(self.W.shape))

    def forward(self, x):
        # print('CapsLayer input shape: {}'.format(x.shape))
        x = x[:, None, ..., None]  # x.unsqueeze(1).unsqueeze(4)
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


class CapsNet(nn.Module):
    def __init__(self, hidden_size=256, cap_output_num=16, vocab_size=16,
                 embedding_size=16, dropout_rate=0.5, num_layers=3):
        super(CapsNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(272, hidden_size, num_layers,
                            dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.primary_layer = PrimaryCapsuleLayer()
        self.caps_layer = CapsLayer(out_channels=cap_output_num)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(cap_output_num, hidden_size)  #
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, seq, sig):
        seq_emb = self.embed(seq.long())
        seq_emb = seq_emb.reshape(seq_emb.shape[0], 1, -1)
        sig = sig.reshape(sig.shape[0], 1, -1)
        # print('seq_emb shape: {}'.format(seq_emb.shape))
        # print('sig shape: {}'.format(sig.shape))
        # to(torch.float32) solve RuntimeError:expected scalar type Double but found Float
        x = torch.cat((seq_emb, sig), dim=1).to(torch.float32)
        x = x.lstm(x)
        # seq = self.primary_layer(seq)
        # seq = self.caps_layer(seq)
        # sig = self.primary_layer(sig)
        # sig = self.caps_layer(sig)
        x = self.primary_layer(x)
        x = self.caps_layer(x)
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # print(x.shape)
        x = x.squeeze(1)

        return x, self.softmax(x)


def test_for_caps_net():
    input1 = torch.rand(1, 1, 105)
    input2 = torch.rand(1, 1, 105)

    model = CapsNet()
    # print(model(input1,input2).shape)


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        weight_rank = torch.from_numpy(np.array([1, 1.0])).float()
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
