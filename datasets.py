import os
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x




class my_ImageDataset_HDRplus(Dataset):        
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "google_hdr_plus/input", set1_input_files[i].strip()))
            self.set1_expert_files.append(os.path.join(root, "google_hdr_plus/output", set1_input_files[i].strip().replace('png', 'jpg')))

        file = open(os.path.join(root,'val.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "google_hdr_plus/input", test_input_files[i].strip()))
            self.test_expert_files.append(os.path.join(root, "google_hdr_plus/output", test_input_files[i].strip().replace('png', 'jpg')))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)

        if self.mode == "train":

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        # =============================== shape: 640*480 ===============================
        _, h, w = img_input.shape
        if h>w:
            img_input = img_input.permute(0,2,1).contiguous()
            img_exptC = img_exptC.permute(0,2,1).contiguous()
        #===============================================================================#

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)