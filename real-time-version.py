# -*- coding: utf-8 -*-
'''
利用opencv获取摄像头图像，并利用pytorch进行风格迁移
'''
import shutil

# @Time    : 2023/11/14 0:25
# @Author  : LINYANZHEN
# @File    : real-time-version.py


import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default="input/content/aiyinsitan.jpg",
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default="input/style/bijiasuo.jpeg",
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')

parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()


class RealTimeTransfer():
    def __init__(self):

        self.output = None
        self.style_image_path = "input/style/bijiasuo.jpeg"
        self.content_size = 256
        self.style_size = 256
        self.crop = 'store_true'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_flag = False
        # 是否保存
        self.save = True
        # 保存路径
        self.save_dir = "output"
        # 保存名字
        self.save_name = "test"
        self.network = self.prepare_model()

    def test_transform(self, size, crop):
        transform_list = []

        if size != 0:
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def style_transform(self, h, w):
        k = (h, w)
        size = int(np.max(k))
        print(type(size))
        transform_list = []
        transform_list.append(transforms.CenterCrop((h, w)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def content_transform(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def prepare_model(self):

        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load(args.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:44])

        decoder = StyTR.decoder
        Trans = transformer.Transformer()
        embedding = StyTR.PatchEmbed()

        decoder.eval()
        Trans.eval()
        vgg.eval()
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        state_dict = torch.load(args.decoder_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        decoder.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(args.Trans_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        Trans.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(args.embedding_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        embedding.load_state_dict(new_state_dict)

        network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
        network.eval()
        network.to(self.device)
        return network

    def style_transfer(self):
        # 启动摄像头
        count = 0
        cap = cv2.VideoCapture(0)
        self.stop_flag = False
        if self.save:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            if os.path.exists(os.path.join(self.save_dir, self.save_name)):
                shutil.rmtree(os.path.join(self.save_dir, self.save_name))
            os.mkdir(os.path.join(self.save_dir, self.save_name))
        while not self.stop_flag:
            # 从摄像头读取一帧图像
            ret, frame = cap.read()
            if not ret:
                break

            # 将图像从OpenCV的BGR格式转换为PIL的RGB格式
            content_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content_image = Image.fromarray(content_image)

            content_tf = self.test_transform(self.content_size, self.crop)
            style_tf = self.test_transform(self.style_size, self.crop)

            # 图像预处理
            # content_tf = content_transform()
            content = content_tf(content_image)

            h, w, c = np.shape(content)
            # style_tf = style_transform(h, w)
            style = style_tf(Image.open(self.style_image_path).convert("RGB"))

            style = style.to(self.device).unsqueeze(0)
            content = content.to(self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.network(content, style)
            output = output[0][0].permute(1, 2, 0).cpu().numpy()
            cv2.imshow('result', output)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                break
            if self.save:
                image_name = os.path.join(self.save_dir, self.save_name, str(count) + ".jpeg")
                output = (output * 255).astype('uint8')
                cv2.imwrite(image_name, output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                count += 1
        # 结束后释放摄像头
        cap.release()

    def img2gif(self):
        frames = []
        image_names = os.listdir(os.path.join(self.save_dir, self.save_name))
        for image in sorted(image_names, key=lambda name: int(''.join(i for i in name if i.isdigit()))):
            frames.append(Image.open(os.path.join(self.save_dir, self.save_name) + '/' + image))
        frames[0].save(self.save_name + '.gif', format='GIF', append_images=frames[1:], save_all=True, duration=80,
                       loop=0)


if __name__ == '__main__':
    model = RealTimeTransfer()
    model.style_transfer()
    cv2.destroyAllWindows()
