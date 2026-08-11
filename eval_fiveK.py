import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from math import exp
import math
from my_models import *
import torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF
from iqa_utils import calculate_ssim    

num_bins_Y = 512
num_bins_RGB = 256
num_bins_RGB_adjusted = 256

def get_lut_classifier_pth(model_dir):
    assert os.path.isdir(model_dir), f"{model_dir} is not a real directory"
    rgb1d_lut_fp = os.path.join(model_dir, "RGB1d_LUTs.pth")
    lut_fp = os.path.join(model_dir, "LUTs.pth")
    ms1d_lut_fp = os.path.join(model_dir, "ms1d_LUTs.pth")
    ms_lut_fp = os.path.join(model_dir, "ms_LUTs.pth")
    classifier_fp = os.path.join(model_dir, "classifier.pth")
    ms_classifier_fp = os.path.join(model_dir, "ms_classifier.pth")

    return classifier_fp, rgb1d_lut_fp, lut_fp, ms_classifier_fp, ms1d_lut_fp, ms_lut_fp

parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", type=str, default="./dataset", help="directory of image")
parser.add_argument("--model_dir", type=str, default="./saved_models/mit_adobe_fiveK", help="directory of pretrained models")		
parser.add_argument("--output_dir", type=str, default="output_fiveK", help="directory to save results")  
opt = parser.parse_args()
window_size = 7
eps=0.0000001

os.makedirs(opt.output_dir, exist_ok=True)

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
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

if cuda:
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
    criterion_pixelwise.cuda()

# ============================================== my functions ==============================================#
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
    max_val1,_ = torch.max(struct, dim=1, keepdim=True)         
    max_val2,_ = torch.max(max_val1, dim=2, keepdim=True)
    max_val3,_ = torch.max(max_val2, dim=3, keepdim=True)

    struct_norm = (struct - min_val3) / (max_val3 - min_val3)
    return struct,struct_norm, mu, sigma
#=============================================================================================================

# Load pretrained models
classifier_fp, rgb1d_lut_fp, lut_fp, ms_classifier_fp, ms1d_lut_fp, ms_lut_fp = get_lut_classifier_pth(opt.model_dir)
print(f'\tPretrain Model used from {opt.model_dir}:\n\t {lut_fp}\n\t {ms_classifier_fp}')
RGB1d_LUTs = torch.load(rgb1d_lut_fp, map_location = device)
LUTs = torch.load(lut_fp, map_location = device)
ms1d_LUTs = torch.load(ms1d_lut_fp, map_location = device)
ms_LUTs = torch.load(ms_lut_fp, map_location = device)
R_LUT0.load_state_dict(RGB1d_LUTs["R0"])
R_LUT1.load_state_dict(RGB1d_LUTs["R1"])
R_LUT2.load_state_dict(RGB1d_LUTs["R2"])
G_LUT0.load_state_dict(RGB1d_LUTs["G0"])
G_LUT1.load_state_dict(RGB1d_LUTs["G1"])
G_LUT2.load_state_dict(RGB1d_LUTs["G2"])
B_LUT0.load_state_dict(RGB1d_LUTs["B0"])
B_LUT1.load_state_dict(RGB1d_LUTs["B1"])
B_LUT2.load_state_dict(RGB1d_LUTs["B2"])
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
mu_LUT0.load_state_dict(ms1d_LUTs["mu0"])
mu_LUT1.load_state_dict(ms1d_LUTs["mu1"])
mu_LUT2.load_state_dict(ms1d_LUTs["mu2"])
sigma_LUT0.load_state_dict(ms1d_LUTs["sigma0"])
sigma_LUT1.load_state_dict(ms1d_LUTs["sigma1"])
sigma_LUT2.load_state_dict(ms1d_LUTs["sigma2"])
ms_LUT0.load_state_dict(ms_LUTs["0"])
ms_LUT1.load_state_dict(ms_LUTs["1"])
ms_LUT2.load_state_dict(ms_LUTs["2"])
classifier.load_state_dict(torch.load(classifier_fp, map_location = device))
classifier.eval()
ms_classifier.load_state_dict(torch.load(ms_classifier_fp, map_location = device))
ms_classifier.eval()

def generator_eval(img):
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
        ms_combine_A[b,:,:,:] = ms_pred[b,0] * ms_gen_A0[b,:,:,:] + ms_pred[b,1] * ms_gen_A1[b,:,:,:] + ms_pred[b,2] * ms_gen_A2[b,:,:,:] #+ pred[b,3] * gen_A3[b,:,:,:] + pred[b,4] * gen_A4[b,:,:,:]

    # recover RGB image using gray image
    fake_mu = ms_combine_A[:,0,:,:].unsqueeze(1)
    fake_sigma = ms_combine_A[:,1,:,:].unsqueeze(1)
    fake_gray = fake_mu + fake_sigma * mscn
    s = s.unsqueeze(2).unsqueeze(3).expand_as(combine_A)
    combine_A_div = (combine_A / torch.mean(combine_A + eps, dim=1, keepdim=True)) **s.expand_as(combine_A)
    output_first_gray = fake_gray.expand_as(combine_A_div)
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

    return output_first_gray, output_2DLUT, output

avg_psnr = 0
avg_ssim = 0
criterion_pixelwise = torch.nn.MSELoss()
criterion_pixelwise.cuda()
file = open(os.path.join(opt.image_dir,'test_fiveK.txt'),'r')
test_input_files = sorted(file.readlines())
dataset_len = len(test_input_files)
for i in range(len(test_input_files)):
    image_path = test_input_files[i].strip()
    # ----------
    #  test
    # ----------
    # read image and transform to tensor
    #fivek
    img = cv2.imread(os.path.join(opt.image_dir, "mit_adobe_fiveK/input/PNG/480p_16bits_XYZ_WB", image_path)+".png", -1)
    gt = Image.open(os.path.join(opt.image_dir, "mit_adobe_fiveK/expertC/JPG/480p", image_path)+".jpg")
    gt = TF.to_tensor(gt).to(device)
    img = np.array(img)
    img = TF_x.to_tensor(img).type(Tensor)
    img = img.unsqueeze(0)

    # generate image
    ms_result_gray, ms_result, result = generator_eval(img)

    # save image
    # final result
    ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    output_im_path = f'{opt.output_dir}/{image_path.split("/")[-1]}'+".png"
    print(f"using model {lut_fp} output image to {output_im_path}")
    im.save(output_im_path, quality=95)

    fake = result
    real = torch.round(gt*255)
    mse = criterion_pixelwise(fake, real)
    psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
    avg_psnr += psnr

    ssim = calculate_ssim(fake.detach().squeeze(0).permute(1,2,0), real.permute(1,2,0), test_y_channel=False)
    avg_ssim += ssim


psnr = avg_psnr / dataset_len
print("PSNR on the test set: ", psnr)
ssim = avg_ssim / dataset_len
print("SSIM on the test set: ", ssim)