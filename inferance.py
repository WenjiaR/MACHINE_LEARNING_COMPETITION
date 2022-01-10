'''
    请大家自行实现测试代码，注意提交格式
'''
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset import Food_LT
from model import resnet34
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger
import config as cfg
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepareTestData(src_path, out_path):
    """
    构建测试集路径txt
    """
    fo = open(out_path, 'w')
    files = os.listdir(src_path)
    for file in files:
        image_path = '/' + src_path + '/' + file
        info = image_path + '\t' + '\n'
        fo.write(info)
    fo.close()

class LT_Dataset_Test(Dataset):
    """
    构建测试数据集
    """
    num_classes = cfg.num_classes

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.transform = transform
        with open(root+txt) as f:
            for line in f:
                path = root + line.split()[0]
                self.img_path.append(path)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path


class Food_LT(object):
    def __init__(self, distributed, root="", num_works=40):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        test_txt = "/data/food/test.txt"

        test_dataset = LT_Dataset_Test(root, test_txt, transform=transform_test)

        # self.test_num_list = test_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None

        self.test_instance = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            num_workers=num_works, pin_memory=True)

def test(test_loader, model, device):
    objfile = open("./data/result.txt",'w')
    model.eval()
    for i, (images, path) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda(cfg.gpu, non_blocking=True)
            output = model(images)
            topv, topi = output.topk(1)
            id = path[0].split('/')[7]
            info = id + ", " + str(topi.item()) + "\n"
            objfile.write(info)


# 将训练集所有图片路径整理到一个txt文件中
src_path = "data/food/test/test"
out_path = "data/food/test.txt"
prepareTestData(src_path, out_path)


model_dir = cfg.root
url = model_dir + '/ckpt/model_best.pth.tar'
model_eval = resnet34()
# model_eval.load_state_dict(torch.load(url, map_location='cuda:0'))

checkpoints = torch.load(url)  #modelpath是你要加载训练好的模型文件地址
model_eval.load_state_dict(checkpoints['state_dict_model'])


dataset = Food_LT(False, root=cfg.root, num_works=0)
test_loader = dataset.test_instance
test(test_loader, model_eval.to(device), device)