import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
import pretrainedmodels
from pretrainedmodels import utils
import h5py
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from PIL import Image
model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

class google_load(object):
    def __init__(self):
        self.t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

    def get(self, img_path):
        img = Image.open(img_path)
        img = self.t(img)
        return img


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True

        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']), strict=False)

        return model

    return GoogLeNet(**kwargs)



class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes=1000, transform_input=False, init_weights=True,
                 blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14


        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14


        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        feature = x.clone()
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, feature


    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, feature = self._forward(x)
        return feature


class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def extract_feats(params, model, load_image_fn, C, H, W):
    model.eval()

    frames_path_list = glob.glob(os.path.join(params['frame_path'], '*'))
    if not params['not_extract_feat']: 
        db = h5py.File(params['feat_dir'], 'a')
    if params['extract_logit']: 
        db2 = h5py.File(params['logit_dir'], 'a')
    

    for frames_dst in tqdm(frames_path_list):
        video_id = frames_dst.split('/')[-1]
        if int(video_id[5:]) > 10000: continue
        if (not params['not_extract_feat'] and video_id in db.keys()) or (params['extract_logit'] and video_id in db2.keys()):
            continue
        
        image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))

        if params['k']: 
            images = torch.zeros((params['k'], C, H, W))
            bound = [int(i) for i in np.linspace(0, len(image_list), params['k']+1)]
            for i in range(params['k']):
                idx = (bound[i] + bound[i+1]) // 2
                if params['model'] == 'googlenet':
                    images[i] = load_image_fn.get(image_list[idx])
                else:
                    images[i] = load_image_fn(image_list[idx])
        else:
            images = torch.zeros((len(image_list), C, H, W))
            for i, image_path in enumerate(image_list):
                images[i] = load_image_fn(image_path)

        with torch.no_grad():
            '''
            feats = model.features(images.cuda())
            logits = model.logits(feats)
            '''
            feats = logits = model(images.cuda())
            
        feats = feats.squeeze().cpu().numpy()
        logits = logits.squeeze().cpu().numpy()

        tqdm.write('%s: %s %s' % (video_id, str(feats.shape), str(logits.shape)))

        if not params['not_extract_feat']: 
            db[video_id] = feats
        if params['extract_logit']: 
            db2[video_id] = logits

    if not params['not_extract_feat']: 
        db.close()
    if params['extract_logit']:       
        db2.close()  

def test_latency(params, model, load_image_fn, C, H, W):
    assert params['test_latency'] > 0
    import time

    model.eval()
    frames_path_list = glob.glob(os.path.join(params['frame_path'], '*'))[:params['test_latency']]
    n_frames = 8
    total_time = 0
    for frames_dst in tqdm(frames_path_list):
        video_id = frames_dst.split('/')[-1]
        image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))
        images = torch.zeros((n_frames, C, H, W))
        bound = [int(i) for i in np.linspace(0, len(image_list), n_frames+1)]
        for i in range(n_frames):
            idx = (bound[i] + bound[i+1]) // 2
            if params['model'] == 'googlenet':
                images[i] = load_image_fn.get(image_list[idx])
            else:
                images[i] = load_image_fn(image_list[idx])

        with torch.no_grad():
            start_time = time.time()
            feats = logits = model(images.cuda())
            total_time += (time.time()-start_time)
    print(total_time, total_time/params['test_latency'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True, help='the path to load all the frames')
    parser.add_argument("--feat_path", type=str, required=True, help='the path you want to save the features')
    parser.add_argument("--feat_name", type=str, default='', help='the name of the features file, saved in hdf5 format')
    parser.add_argument("--logit_name", type=str, default='', help='the name of the logits file, saved in hdf5 format')
    
    parser.add_argument("-nef", "--not_extract_feat", default=False, action='store_true')
    parser.add_argument("-el", "--extract_logit", default=False, action='store_true')
    
    parser.add_argument("--gpu", type=str, default='0', help='set CUDA_VISIBLE_DEVICES environment variable')
    parser.add_argument("--model", type=str, default='inceptionresnetv2', help='inceptionresnetv2 | resnet101')
    
    parser.add_argument("--k", type=int, default=60, 
        help='uniformly sample k frames from the existing frames and then extract their features. k=0 will extract all existing frames')
    parser.add_argument("--frame_suffix", type=str, default='jpg')

    parser.add_argument("--test_latency", type=int, default=0)
    
    args = parser.parse_args()
    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

    assert os.path.exists(params['frame_path'])
    if not os.path.exists(params['feat_path']):
        os.makedirs(params['feat_path'])

    if not params['not_extract_feat']: assert params['feat_name']
    if params['extract_logit']: assert params['logit_name']

    params['feat_dir'] = os.path.join(params['feat_path'], params['feat_name'] + ('' if '.hdf5' in params['feat_name'] else '.hdf5'))
    params['logit_dir'] = os.path.join(params['feat_path'], params['logit_name'] + ('' if '.hdf5' in params['logit_name'] else '.hdf5'))

    print('Model: %s' % params['model'])
    print('The extracted features will be saved to --> %s' % params['feat_dir'])

    if params['model'] == 'resnet101':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet101(pretrained='imagenet')
    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif params['model'] == 'resnet18':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet18(pretrained='imagenet')
    elif params['model'] == 'resnet34':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet34(pretrained='imagenet')
    elif params['model'] == 'inceptionresnetv2':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionresnetv2(
            num_classes=1001, pretrained='imagenet+background')
    elif params['model'] == 'googlenet':
        C, H, W = 3, 224, 224
        model = googlenet(pretrained=True)
        print(model)
    else:
        print("doesn't support %s" % (params['model']))

    if params['model'] != 'googlenet':
        load_image_fn = utils.LoadTransformImage(model)
        model.last_linear = utils.Identity() 
    else:
        load_image_fn = google_load()

    model = model.cuda()

    #summary(model, (C, H, W))
    if params['test_latency']:
        test_latency(params, model, load_image_fn, C, H, W)
    else:
        extract_feats(params, model, load_image_fn, C, H, W)

'''
python extract_image_feats_from_frames.py \
--frame_path "/home/yangbang/VideoCaptioning/MSRVTT/all_frames/" \
--feat_path "/home/yangbang/VideoCaptioning/MSRVTT/feats/" \
--feat_name msrvtt_R152 \
--model resnet152 \
--k 60 \
--frame_suffix jpg \
--gpu 2

python extract_image_feats_from_frames.py \
--frame_path "/home/yangbang/VideoCaptioning/Youtube2Text/all_frames/" \
--feat_path "/home/yangbang/VideoCaptioning/Youtube2Text/feats/" \
--feat_name msvd_R152 \
--model resnet152 \
--k 60 \
--frame_suffix jpg \
--gpu 3
'''