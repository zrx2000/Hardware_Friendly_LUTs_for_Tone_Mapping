import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import math
import itertools
import time
import datetime
import sys
from math import exp
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets import *
from my_models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16
from loss_network import vgg_LossNetwork

window_size = 7
num_bins_Y = 512
num_bins_RGB = 256
num_bins_RGB_adjusted = 256
eps = 0.0000001
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--restore", type=bool, default=False, help="restore from pth")
parser.add_argument("--n_epochs", type=int, default=300, help="total number of epochs of training")
parser.add_argument('--dataset_dir', type = str, default = "./dataset", help = "Path to Dataset directory containing sub-directories of train and test of which each also contains sub-direcotries of input and output")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")   
parser.add_argument("--lr", type=float, default=0.005, help="adam: learning rate")        
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_perceptual", type=float, default=0.01)
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")  
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")    
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--output_dir", type=str, default="google_hdr_plus", help="path to save model")
opt = parser.parse_args()
lr = opt.lr

# ============================================== vgg loss network ==============================================
#initialize perceptual loss model
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = vgg_LossNetwork(vgg_model)
loss_network.eval()

# ============================================== MSCN functions ==============================================#
def gaussian(window_size, sigma):   # return a 1-d Gaussian vector given kernel size and sigma
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):  #return a 2-d Gaussian kernel
    _1D_window = gaussian(window_size, window_size/6.0).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

window = create_window(window_size)
if torch.cuda.is_available():
    window = window.cuda()

def mscn_torch(input_img, ksize, c, window):
    if(input_img.shape[1]==3):
        input_img = torch.mean(input_img, dim=1).unsqueeze(1)
    pad = (ksize-1) // 2
    window = window.to(input_img.device)
    mu = F.conv2d(input_img, window, padding=pad, groups=1)   
    mu_sq = mu.pow(2)
    sigma = torch.sqrt(torch.abs(F.conv2d(input_img * input_img, window, padding=pad, groups=1) - mu_sq) + 1e-8)
    struct = (input_img - mu) / (sigma + c)

    min_val1,_ = torch.min(struct, dim=1, keepdim=True)
    min_val2,_ = torch.min(min_val1, dim=2, keepdim=True)
    min_val3,_ = torch.min(min_val2, dim=3, keepdim=True)
    max_val1,_ = torch.min(struct, dim=1, keepdim=True)
    max_val2,_ = torch.min(max_val1, dim=2, keepdim=True)
    max_val3,_ = torch.min(max_val2, dim=3, keepdim=True)

    struct_norm = (struct - min_val3) / (max_val3 - min_val3)
    return struct,struct_norm, mu, sigma
#=============================================================================================================

opt.output_dir = opt.output_dir
print(opt)

os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()    
criterion_smooth_l1 = torch.nn.L1Loss()

# models
# for RGB
R_LUT0 = Generator1DLUT_gamma(dim=513, gamma=1)
R_LUT1 = Generator1DLUT_gamma(dim=513, gamma=2)
R_LUT2 = Generator1DLUT_gamma(dim=513, gamma=0.2)
G_LUT0 = Generator1DLUT_gamma(dim=513, gamma=1)
G_LUT1 = Generator1DLUT_gamma(dim=513, gamma=2)
G_LUT2 = Generator1DLUT_gamma(dim=513, gamma=0.2)
B_LUT0 = Generator1DLUT_gamma(dim=513, gamma=1)
B_LUT1 = Generator1DLUT_gamma(dim=513, gamma=2)
B_LUT2 = Generator1DLUT_gamma(dim=513, gamma=0.2)
LUT0 = Generator3DLUT_gamma(dim=9, gamma=2)   
LUT1 = Generator3DLUT_gamma(dim=9, gamma=1)
LUT2 = Generator3DLUT_gamma(dim=9, gamma=0.2)
classifier = rgb_Classifier(bin_num=num_bins_RGB_adjusted)
# for mu and sigma (ms)
mu_LUT0 = Generator1DLUT_gamma(dim=513, gamma=1) 
mu_LUT1 = Generator1DLUT_gamma(dim=513, gamma=2)
mu_LUT2 = Generator1DLUT_gamma(dim=513, gamma=0.2)
sigma_LUT0 = Generator1DLUT_gamma(dim=513, gamma=1)  
sigma_LUT1 = Generator1DLUT_gamma(dim=513, gamma=2)
sigma_LUT2 = Generator1DLUT_gamma(dim=513, gamma=0.2)
ms_LUT0 = Generator2DLUT_identity(dim=33)    
ms_LUT1 = Generator2DLUT_identity(dim=33)
ms_LUT2 = Generator2DLUT_identity(dim=33)
ms_classifier = ms_Classifier()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if cuda:
    criterion_pixelwise.cuda()
    criterion_smooth_l1.cuda()

    R_LUT0 = R_LUT0.cuda()
    R_LUT1 = R_LUT1.cuda()
    R_LUT2 = R_LUT2.cuda()
    G_LUT0 = G_LUT0.cuda()
    G_LUT1 = G_LUT1.cuda()
    G_LUT2 = G_LUT2.cuda()
    B_LUT0 = B_LUT0.cuda()
    B_LUT1 = B_LUT1.cuda()
    B_LUT2 = B_LUT2.cuda()
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    mu_LUT0     = mu_LUT0.cuda()
    mu_LUT1     = mu_LUT1.cuda()
    mu_LUT2     = mu_LUT2.cuda()
    sigma_LUT0     = sigma_LUT0.cuda()
    sigma_LUT1     = sigma_LUT1.cuda()
    sigma_LUT2     = sigma_LUT2.cuda()
    ms_LUT0 = ms_LUT0.cuda()
    ms_LUT1 = ms_LUT1.cuda()
    ms_LUT2 = ms_LUT2.cuda()
    ms_classifier = ms_classifier.cuda()

# Load pretrained models  
if opt.restore == True:
    RGB1d_LUTs = torch.load("./saved_models/google_hdr_plus/RGB1d_LUTs.pth")
    R_LUT0.load_state_dict(RGB1d_LUTs["R0"])
    R_LUT1.load_state_dict(RGB1d_LUTs["R1"])
    R_LUT2.load_state_dict(RGB1d_LUTs["R2"])
    G_LUT0.load_state_dict(RGB1d_LUTs["G0"])
    G_LUT1.load_state_dict(RGB1d_LUTs["G1"])
    G_LUT2.load_state_dict(RGB1d_LUTs["G2"])
    B_LUT0.load_state_dict(RGB1d_LUTs["B0"])
    B_LUT1.load_state_dict(RGB1d_LUTs["B1"])
    B_LUT2.load_state_dict(RGB1d_LUTs["B2"])
    LUTs = torch.load("./saved_models/google_hdr_plus/LUTs.pth")
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    classifier.load_state_dict(torch.load("./saved_models/google_hdr_plus/classifier.pth"))

    ms1d_LUTs = torch.load("./saved_models/google_hdr_plus/ms1d_LUTs.pth")
    mu_LUT0.load_state_dict(ms1d_LUTs["mu0"])
    mu_LUT1.load_state_dict(ms1d_LUTs["mu1"])
    mu_LUT2.load_state_dict(ms1d_LUTs["mu2"])
    sigma_LUT0.load_state_dict(ms1d_LUTs["sigma0"])
    sigma_LUT1.load_state_dict(ms1d_LUTs["sigma1"])
    sigma_LUT2.load_state_dict(ms1d_LUTs["sigma2"])
    ms_LUTs = torch.load("./saved_models/google_hdr_plus/ms_LUTs.pth")
    ms_LUT0.load_state_dict(ms_LUTs["0"])
    ms_LUT1.load_state_dict(ms_LUTs["1"])
    ms_LUT2.load_state_dict(ms_LUTs["2"])
    ms_classifier.load_state_dict(torch.load("./saved_models/google_hdr_plus/ms_classifier.pth"))

else:
    # Initialize weights
    classifier.apply(weights_init_normal_classifier)
    ms_classifier.apply(weights_init_normal_classifier)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(
                            R_LUT0.parameters(), R_LUT1.parameters(), R_LUT2.parameters(), G_LUT0.parameters(), G_LUT1.parameters(), G_LUT2.parameters(), B_LUT0.parameters(), B_LUT1.parameters(), B_LUT2.parameters(),
                            classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters(),
                            mu_LUT0.parameters(), mu_LUT1.parameters(), mu_LUT2.parameters(), sigma_LUT0.parameters(), sigma_LUT1.parameters(), sigma_LUT2.parameters(), 
                            ms_classifier.parameters(), ms_LUT0.parameters(), ms_LUT1.parameters(), ms_LUT2.parameters()), 
                            lr=opt.lr, betas=(opt.b1, opt.b2)) 


dataloader = DataLoader(
    my_ImageDataset_HDRplus(opt.dataset_dir, mode = "train"),    
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

psnr_dataloader = DataLoader(
    my_ImageDataset_HDRplus(opt.dataset_dir, mode = "test"),    
    batch_size=1,
    shuffle=False,
    num_workers=1
)

def generator_train(img):
    b,_,h,w = img.shape
    num_pixels = h*w

    #================================================ MS LUT
    combine_A = img
    combine_A = torch.clamp(combine_A, min=eps, max=1)        
    combine_A_Y = torch.mean(combine_A, dim=1).unsqueeze(1)
    hist_Y = torch.empty(b, num_bins_Y).to(device)

    # adjust mu and sigma
    mscn, mscn_norm, mu, sigma = mscn_torch(combine_A, window_size, eps, window)
    for j in range(b):
        Y_data = combine_A_Y[j,:,:,:].view(-1)
        hist_Y[j,:] = torch.histc(Y_data, bins=num_bins_Y, min=0, max=1)
    hist_Y = hist_Y / num_pixels
    mu1d_pred0, mu1d_pred1, mu1d_pred2, sigma1d_pred0, sigma1d_pred1, sigma1d_pred2, ms_pred, s = ms_classifier(hist_Y)

    mu_gen_A0 = mu_LUT0(mu)    
    mu_gen_A1 = mu_LUT1(mu)
    mu_gen_A2 = mu_LUT2(mu)
    sigma_gen_A0 = sigma_LUT0(sigma)    
    sigma_gen_A1 = sigma_LUT1(sigma)
    sigma_gen_A2 = sigma_LUT2(sigma)
    mu_adjust0 = torch.empty(b,1,h,w).to(device)
    mu_adjust1 = torch.empty(b,1,h,w).to(device)
    mu_adjust2 = torch.empty(b,1,h,w).to(device)
    sigma_adjust0 = torch.empty(b,1,h,w).to(device)
    sigma_adjust1 = torch.empty(b,1,h,w).to(device)
    sigma_adjust2 = torch.empty(b,1,h,w).to(device)
    for b in range(img.size(0)):
        mu_adjust_temp0 = mu1d_pred0[b,0] * mu_gen_A0[b,:,:,:] + mu1d_pred0[b,1] * mu_gen_A1[b,:,:,:] + mu1d_pred0[b,2] * mu_gen_A2[b,:,:,:]
        mu_adjust_temp0 = torch.clamp(mu_adjust_temp0, 0, 1)
        mu_adjust_temp1 = mu1d_pred1[b,0] * mu_gen_A0[b,:,:,:] + mu1d_pred1[b,1] * mu_gen_A1[b,:,:,:] + mu1d_pred1[b,2] * mu_gen_A2[b,:,:,:]
        mu_adjust_temp1 = torch.clamp(mu_adjust_temp1, 0, 1)
        mu_adjust_temp2 = mu1d_pred2[b,0] * mu_gen_A0[b,:,:,:] + mu1d_pred2[b,1] * mu_gen_A1[b,:,:,:] + mu1d_pred2[b,2] * mu_gen_A2[b,:,:,:]
        mu_adjust_temp2 = torch.clamp(mu_adjust_temp2, 0, 1)
        sigma_adjust_temp0 = sigma1d_pred0[b,0] * sigma_gen_A0[b,:,:,:] + sigma1d_pred0[b,1] * sigma_gen_A1[b,:,:,:] + sigma1d_pred0[b,2] * sigma_gen_A2[b,:,:,:]
        sigma_adjust_temp0 = torch.clamp(sigma_adjust_temp0, 0, 1)
        sigma_adjust_temp1 = sigma1d_pred1[b,0] * sigma_gen_A0[b,:,:,:] + sigma1d_pred1[b,1] * sigma_gen_A1[b,:,:,:] + sigma1d_pred1[b,2] * sigma_gen_A2[b,:,:,:]
        sigma_adjust_temp1 = torch.clamp(sigma_adjust_temp1, 0, 1)
        sigma_adjust_temp2 = sigma1d_pred2[b,0] * sigma_gen_A0[b,:,:,:] + sigma1d_pred2[b,1] * sigma_gen_A1[b,:,:,:] + sigma1d_pred2[b,2] * sigma_gen_A2[b,:,:,:]
        sigma_adjust_temp2 = torch.clamp(sigma_adjust_temp2, 0, 1)
        mu_adjust0[b,:,:,:] = mu_adjust_temp0
        mu_adjust1[b,:,:,:] = mu_adjust_temp1
        mu_adjust2[b,:,:,:] = mu_adjust_temp2
        sigma_adjust0[b,:,:,:] = sigma_adjust_temp0
        sigma_adjust1[b,:,:,:] = sigma_adjust_temp1
        sigma_adjust2[b,:,:,:] = sigma_adjust_temp2

    input_ms0 = torch.cat([mu_adjust0, sigma_adjust0], dim=1)
    input_ms1 = torch.cat([mu_adjust1, sigma_adjust1], dim=1)
    input_ms2 = torch.cat([mu_adjust2, sigma_adjust2], dim=1)
    ms_gen_A0 = ms_LUT0(input_ms0)    
    ms_gen_A1 = ms_LUT1(input_ms1)
    ms_gen_A2 = ms_LUT2(input_ms2)

    ms_combine_A = input_ms0.new(input_ms0.size())   
    for b in range(img.size(0)):
        ms_combine_A[b,:,:,:] = ms_pred[b,0] * ms_gen_A0[b,:,:,:] + ms_pred[b,1] * ms_gen_A1[b,:,:,:] + ms_pred[b,2] * ms_gen_A2[b,:,:,:] 

    # recover RGB image using gray image
    fake_mu = ms_combine_A[:,0,:,:].unsqueeze(1)
    fake_sigma = ms_combine_A[:,1,:,:].unsqueeze(1)
    fake_gray = fake_mu + fake_sigma * mscn
    s = s.unsqueeze(2).unsqueeze(3).expand_as(combine_A)
    combine_A_div = (combine_A / torch.mean(combine_A + eps, dim=1, keepdim=True)) **s.expand_as(combine_A)
    output_2DLUT = combine_A_div * fake_gray.expand_as(combine_A_div)

    #================================================ RGB LUT
    b,_,h,w = img.shape
    hist_R = torch.empty(b, num_bins_RGB).to(device)
    hist_G = torch.empty(b, num_bins_RGB).to(device)
    hist_B = torch.empty(b, num_bins_RGB).to(device)

    for j in range(b):
        hist_R[j,:] = torch.histc(output_2DLUT[j,2,:,:].view(-1), bins=num_bins_RGB, min=0, max=1)
        hist_G[j,:] = torch.histc(output_2DLUT[j,1,:,:].view(-1), bins=num_bins_RGB, min=0, max=1)
        hist_B[j,:] = torch.histc(output_2DLUT[j,0,:,:].view(-1), bins=num_bins_RGB, min=0, max=1)
    hist_R = hist_R / num_pixels
    hist_G = hist_G / num_pixels
    hist_B = hist_B / num_pixels
    R1d_pred0, R1d_pred1, R1d_pred2, G1d_pred0, G1d_pred1, G1d_pred2, B1d_pred0, B1d_pred1, B1d_pred2, rgb_pred = classifier(hist_R, hist_G, hist_B)
    R_gen_A0 = R_LUT0(output_2DLUT[:,2,:,:].unsqueeze(1))
    R_gen_A1 = R_LUT1(output_2DLUT[:,2,:,:].unsqueeze(1))
    R_gen_A2 = R_LUT2(output_2DLUT[:,2,:,:].unsqueeze(1))
    G_gen_A0 = G_LUT0(output_2DLUT[:,1,:,:].unsqueeze(1))
    G_gen_A1 = G_LUT1(output_2DLUT[:,1,:,:].unsqueeze(1))
    G_gen_A2 = G_LUT2(output_2DLUT[:,1,:,:].unsqueeze(1))
    B_gen_A0 = B_LUT0(output_2DLUT[:,0,:,:].unsqueeze(1))
    B_gen_A1 = B_LUT1(output_2DLUT[:,0,:,:].unsqueeze(1))
    B_gen_A2 = B_LUT2(output_2DLUT[:,0,:,:].unsqueeze(1))
    R_adjust0 = R1d_pred0[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A0 + R1d_pred0[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A1 + R1d_pred0[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A2
    R_adjust1 = R1d_pred1[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A0 + R1d_pred1[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A1 + R1d_pred1[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A2
    R_adjust2 = R1d_pred2[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A0 + R1d_pred2[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A1 + R1d_pred2[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * R_gen_A2
    G_adjust0 = G1d_pred0[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A0 + G1d_pred0[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A1 + G1d_pred0[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A2
    G_adjust1 = G1d_pred1[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A0 + G1d_pred1[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A1 + G1d_pred1[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A2
    G_adjust2 = G1d_pred2[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A0 + G1d_pred2[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A1 + G1d_pred2[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * G_gen_A2
    B_adjust0 = B1d_pred0[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A0 + B1d_pred0[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A1 + B1d_pred0[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A2
    B_adjust1 = B1d_pred1[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A0 + B1d_pred1[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A1 + B1d_pred1[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A2
    B_adjust2 = B1d_pred2[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A0 + B1d_pred2[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A1 + B1d_pred2[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * B_gen_A2
    RGB_adjusted0 = torch.cat([B_adjust0, G_adjust0, R_adjust0], dim=1)
    RGB_adjusted0 = torch.clamp(RGB_adjusted0, min=eps, max=1)
    RGB_adjusted1 = torch.cat([B_adjust1, G_adjust1, R_adjust1], dim=1)
    RGB_adjusted1 = torch.clamp(RGB_adjusted1, min=eps, max=1)
    RGB_adjusted2 = torch.cat([B_adjust2, G_adjust2, R_adjust2], dim=1)
    RGB_adjusted2 = torch.clamp(RGB_adjusted2, min=eps, max=1)

    gen_A0 = LUT0(RGB_adjusted0)
    gen_A1 = LUT1(RGB_adjusted1)
    gen_A2 = LUT2(RGB_adjusted2)
    gen_RGB = rgb_pred[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * gen_A0 + rgb_pred[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * gen_A1 + rgb_pred[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * gen_A2

    output = gen_RGB

    return output_2DLUT, output

def calculate_psnr():
    classifier.eval()
    ms_classifier.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        _, fake_B = generator_train(real_A)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr
    classifier.train()
    ms_classifier.train()

    return avg_psnr/ len(psnr_dataloader)


# ----------
#  Training
# ----------

prev_time = time.time()
max_psnr = 0
max_epoch = 0
for epoch in range(opt.epoch, opt.n_epochs):
    if (epoch == 30) or (epoch == 100):     
        lr = lr /2.
        optimizer_G = torch.optim.Adam(itertools.chain(
                                        R_LUT0.parameters(), R_LUT1.parameters(), R_LUT2.parameters(), G_LUT0.parameters(), G_LUT1.parameters(), G_LUT2.parameters(), B_LUT0.parameters(), B_LUT1.parameters(), B_LUT2.parameters(),
                                        classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters(),
                                        mu_LUT0.parameters(), mu_LUT1.parameters(), mu_LUT2.parameters(), sigma_LUT0.parameters(), sigma_LUT1.parameters(), sigma_LUT2.parameters(), 
                                        ms_classifier.parameters(), ms_LUT0.parameters(), ms_LUT1.parameters(), ms_LUT2.parameters()), 
                                           lr=lr, betas=(opt.b1, opt.b2))
    mse_avg = 0
    perc_avg = 0
    psnr_avg = 0
    #classifier.train()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        _, fake_B = generator_train(real_A)

        # Pixel-wise loss
        mse = criterion_pixelwise(fake_B, real_B)
        perceptual_loss = loss_network(fake_B, real_B)
        l1_loss = criterion_smooth_l1(fake_B, real_B)

        loss = l1_loss + opt.lambda_perceptual * perceptual_loss

        psnr_avg += 10 * math.log10(1 / mse.item())

        mse_avg += mse.item()
        perc_avg += perceptual_loss.item()

        loss.backward()

        optimizer_G.step()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
			"\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f, perceptual: %f] ETA: %s"
			% (epoch,opt.n_epochs,i,len(dataloader),psnr_avg / (i+1), perc_avg / (i+1), time_left,
			)
		)

    avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
        RGB1d_LUTs = {"R0": R_LUT0.state_dict(),"R1": R_LUT1.state_dict(),"R2": R_LUT2.state_dict(),
                    "G0": G_LUT0.state_dict(),"G1": G_LUT1.state_dict(),"G2": G_LUT2.state_dict(),
                    "B0": B_LUT0.state_dict(),"B1": B_LUT1.state_dict(),"B2": B_LUT2.state_dict()}
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(), "2": LUT2.state_dict()} 
        ms1d_LUTs = {"mu0": mu_LUT0.state_dict(),"mu1": mu_LUT1.state_dict(),"mu2": mu_LUT2.state_dict(),
                    "sigma0": sigma_LUT0.state_dict(),"sigma1": sigma_LUT1.state_dict(),"sigma2": sigma_LUT2.state_dict()}
        ms_LUTs = {"0": ms_LUT0.state_dict(),"1": ms_LUT1.state_dict(),"2": ms_LUT2.state_dict()}
        torch.save(RGB1d_LUTs, "saved_models/%s/RGB1d_LUTs_best.pth" % (opt.output_dir))
        torch.save(LUTs, "saved_models/%s/LUTs_best.pth" % (opt.output_dir))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier_best.pth" % (opt.output_dir))
        torch.save(ms1d_LUTs, "saved_models/%s/ms1d_LUTs_best.pth" % (opt.output_dir))
        torch.save(ms_LUTs, "saved_models/%s/ms_LUTs_best.pth" % (opt.output_dir))
        torch.save(ms_classifier.state_dict(), "saved_models/%s/ms_classifier_best.pth" % (opt.output_dir))
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))


    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        RGB1d_LUTs = {"R0": R_LUT0.state_dict(),"R1": R_LUT1.state_dict(),"R2": R_LUT2.state_dict(),
                    "G0": G_LUT0.state_dict(),"G1": G_LUT1.state_dict(),"G2": G_LUT2.state_dict(),
                    "B0": B_LUT0.state_dict(),"B1": B_LUT1.state_dict(),"B2": B_LUT2.state_dict()}
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(), "2": LUT2.state_dict()} 
        ms1d_LUTs = {"mu0": mu_LUT0.state_dict(),"mu1": mu_LUT1.state_dict(),"mu2": mu_LUT2.state_dict(),
                    "sigma0": sigma_LUT0.state_dict(),"sigma1": sigma_LUT1.state_dict(),"sigma2": sigma_LUT2.state_dict()}
        ms_LUTs = {"0": ms_LUT0.state_dict(),"1": ms_LUT1.state_dict(),"2": ms_LUT2.state_dict()}
        torch.save(RGB1d_LUTs, "saved_models/%s/RGB1d_LUTs.pth" % (opt.output_dir))
        torch.save(LUTs, "saved_models/%s/LUTs.pth" % (opt.output_dir))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier.pth" % (opt.output_dir))
        torch.save(ms1d_LUTs, "saved_models/%s/ms1d_LUTs.pth" % (opt.output_dir))
        torch.save(ms_LUTs, "saved_models/%s/ms_LUTs.pth" % (opt.output_dir))
        torch.save(ms_classifier.state_dict(), "saved_models/%s/ms_classifier.pth" % (opt.output_dir))
        file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        file.close()