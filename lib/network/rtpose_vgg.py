"""CPM Pytorch Implementation"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from .resnet50_fpn_model import ResNet, Bottleneck
from model.dht import DHT_Layer, DHT



def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def make_resnet_block(param):

    return ResNet(Bottleneck, param, include_top=False)

def get_model(trunk, args):

    blocks = {}
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3': [512, 512, 3, 1, 1]},
                  {'conv4_4': [512, 512, 3, 1, 1]},
                  # {'pool4_stage1': [2, 2, 0]},
                  # {'conv5_1': [512, 512, 3, 1, 1]},
                  # {'conv5_2': [512, 512, 3, 1, 1]},
                  # {'conv5_3': [512, 512, 3, 1, 1]},
                  # {'conv5_4': [512, 512, 3, 1, 1]},
                  # {'pool5_stage1': [2, 2, 0]},
                  # {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  # {'conv4_4_CPM': [256, 128, 3, 1, 1]}
                  ]

    block00 = [{'conv4_3_CPM': [512, 256, 3, 1, 1]},
              {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 16, 1, 1, 0]}]

    for i in range(2, 3):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [144, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 16, 1, 1, 0]}
        ]


    models = {}

    if trunk == 'vgg19':
        print("Building VGG19")
        models['block0'] = make_vgg19_block(block0)

    elif trunk == 'resnet50':
        print("Building ResNet50")
        models['block0'] = make_resnet_block([3, 4])

    elif trunk == 'resnet101':
        print("Building ResNet101")
        models['block0'] = make_resnet_block([3, 4])

    elif trunk == 'resnet152':
        print("Building ResNet152")
        models['block0'] = make_resnet_block([3, 8])


    models['block00'] = make_vgg19_block(block00)

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    class rtpose_model(nn.Module):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model00 = model_dict['block00']
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.DHT_layer2 = DHT_Layer(16, 8, numAngle=100, numRho=50)
            self.lastlayer = nn.Sequential(
                nn.Conv2d(8, 1, 1)
            )

            self._initialize_weights_norm()

        def forward(self, x):
            out1 = self.model0(x)
            out1 = self.model00(out1)
            out_scales = {}  # 存储单尺度特征图
            for i in range(args.num_fpn_scale):
                out1_1 = self.model1_1(out1)
                out2 = torch.cat([out1_1, out1], 1)

                out2_1 = self.model2_1(out2)
                # out2_2 = torch.abs(out2_1)
                out2_3 = self.DHT_layer2(out2_1)
                out2_3 = self.lastlayer(out2_3)
                out_scales[i] = [out1_1, out2_1, out2_3]

            return out_scales

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)

            # last layer of these block don't have Relu
            init.normal_(self.model1_1[12].weight, std=0.01)
            init.normal_(self.model2_1[18].weight, std=0.01)

    model = rtpose_model(models)
    return model

def use_vgg(model):

    url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    vgg_state_dict = model_zoo.load_url(url)
    vgg_keys = vgg_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(24):
        weights_load[list(model.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model')





def use_resnet(model):

    # url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    resnet_state_dict = model_zoo.load_url(url)
    vgg_keys = resnet_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    keys = list(model.state_dict().keys())
    for i in range(len(keys) - 1, -1, -1):
        if keys[i][-7:] == 'tracked':
            keys.pop(i)
    for i in range(100):
        weights_load[keys[i]] = resnet_state_dict[list(vgg_keys)[i]]
    state = model.state_dict()
    for key in list(state.keys()):
        if key[-7:] == 'tracked':
            del state[key]
    state.update(weights_load)
    model.load_state_dict(state, False)
    print('load imagenet pretrained model')

