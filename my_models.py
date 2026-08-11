import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

    
class ms_Classifier(nn.Module):        
    def __init__(self, bin_num=512, lut_num=3):
        super(ms_Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(bin_num, bin_num//8),  
            nn.ReLU(),
            nn.Dropout(p=0.3), 
            nn.Linear(bin_num//8, 3*6+4)    
        )
        self.output_layer = nn.Softmax(dim=1)
        self.last_relu = nn.ReLU()

    def forward(self, hist):       
        weights = self.layers(hist)
        mu1d_pred0 = self.output_layer(weights[:,:3])
        mu1d_pred1 = self.output_layer(weights[:,3:6])
        mu1d_pred2 = self.output_layer(weights[:,6:9])
        sigma1d_pred0 = self.output_layer(weights[:,9:12])
        sigma1d_pred1 = self.output_layer(weights[:,12:15])
        sigma1d_pred2 = self.output_layer(weights[:,15:18])
        ms_pred = self.output_layer(weights[:,18:21])
        return mu1d_pred0, mu1d_pred1, mu1d_pred2, sigma1d_pred0, sigma1d_pred1, sigma1d_pred2, ms_pred, self.last_relu(weights[:,-1]).unsqueeze(1)

    
class rgb_Classifier(nn.Module):        
    def __init__(self, bin_num=256, lut_num=3):
        super(rgb_Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*bin_num, bin_num//4), 
            nn.ReLU(),
            nn.Dropout(p=0.3),  
            nn.Linear(bin_num//4, 3*9+3)    
        )
        self.output_layer = nn.Softmax(dim=1)
        self.last_relu = nn.ReLU()
        self.lut_num = lut_num

    def forward(self, hist_R, hist_G, hist_B):  
        hist = torch.cat([hist_R, hist_G, hist_B], dim=1)     
        weights = self.layers(hist)
        R1d_pred0 = self.output_layer(weights[:,:self.lut_num])
        R1d_pred1 = self.output_layer(weights[:,self.lut_num:2*self.lut_num])
        R1d_pred2 = self.output_layer(weights[:,2*self.lut_num:3*self.lut_num]) 
        G1d_pred0 = self.output_layer(weights[:,3*self.lut_num:4*self.lut_num])
        G1d_pred1 = self.output_layer(weights[:,4*self.lut_num:5*self.lut_num])
        G1d_pred2 = self.output_layer(weights[:,5*self.lut_num:6*self.lut_num]) 
        B1d_pred0 = self.output_layer(weights[:,6*self.lut_num:7*self.lut_num])
        B1d_pred1 = self.output_layer(weights[:,7*self.lut_num:8*self.lut_num])
        B1d_pred2 = self.output_layer(weights[:,8*self.lut_num:9*self.lut_num]) 
        rgb_pred  = self.output_layer(weights[:,27:30]) 
        return R1d_pred0, R1d_pred1, R1d_pred2, G1d_pred0, G1d_pred1, G1d_pred2, B1d_pred0, B1d_pred1, B1d_pred2, rgb_pred

class Generator2DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator2DLUT_identity, self).__init__()
        buffer = np.zeros((2,dim,dim), dtype=np.float32)
        delta = 1. / (dim-1)

        for i in range(0,dim):
            for j in range(0,dim):
                    buffer[0,i,j] = j*delta
                    buffer[1,i,j] = i*delta
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):   
        img = (x - .5) * 2.
        img = img.permute(0, 2, 3, 1)
        bs,_,_,_ = img.shape
        LUT = self.LUT[None].repeat(bs,1,1,1)
        result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)        
        return result


class TV_2D(nn.Module):
    def __init__(self, dim=33):
        super(TV_2D,self).__init__()

        self.weight_sigma = torch.ones(2,dim,dim-1, dtype=torch.float)
        self.weight_sigma[:,:,(0,dim-2)] *= 2.0
        self.weight_mu = torch.ones(2,dim-1,dim, dtype=torch.float)
        self.weight_mu[:,(0,dim-2),:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_sigma = LUT.LUT[:,:,:-1] - LUT.LUT[:,:,1:]
        dif_mu = LUT.LUT[:,:-1,:] - LUT.LUT[:,1:,:]
        tv_sigma = torch.mean(torch.mul((dif_sigma ** 2),self.weight_sigma)) 
        tv_mu = torch.mean(torch.mul((dif_mu ** 2),self.weight_mu)) 

        mn_sigma = torch.mean(self.relu(dif_sigma)) 
        mn_mu = torch.mean(self.relu(dif_mu)) 
        return tv_sigma + tv_mu, mn_sigma + mn_mu

class Generator3DLUT_gamma(nn.Module):       
    def __init__(self, dim=33, gamma=1/2.2):
        super(Generator3DLUT_gamma, self).__init__()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
        interval = 1 / (dim - 1)
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    buffer[0,i,j,k] = float((i * interval) ** gamma)
                    buffer[1,i,j,k] = float((j * interval) ** gamma)
                    buffer[2,i,j,k] = float((k * interval) ** gamma)
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        img = (x - .5) * 2.
        img = img.permute(0, 2, 3, 1)[:, None]
        bs,_,_,_,_ = img.shape
        LUT = self.LUT[None].repeat(bs,1,1,1,1)
        result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
        output = result[:, :, 0]        
        return output

class Generator1DLUT_gamma(nn.Module):       
    def __init__(self, dim=33, gamma=1/2.2):
        super(Generator1DLUT_gamma, self).__init__()
        self.dim = dim
        buffer = np.zeros((1,dim), dtype=np.float32)
        interval = 1. / (dim - 1)
        for i in range(0,dim):
                    buffer[0,i] = float((i * interval) ** gamma)
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        B, C, H, W = x.shape
        x_normalized = (x - 0.5) * 2.0 
        x_flat = x_normalized.reshape(B * C, H, W)
        LUT_expanded = self.LUT.unsqueeze(1).unsqueeze(1)  
        LUT_expanded = LUT_expanded.repeat(B * C, 1, 1, 1)  
        
        grid_x = x_flat  
        grid_y = torch.zeros_like(x_flat)  
        grid = torch.stack([grid_x, grid_y], dim=-1)  
        
        # grid_sample
        # input: [B*C, 1, 1, dim] - 4D tensor
        # grid: [B*C, H, W, 2] - 4D tensor
        result = F.grid_sample(
            LUT_expanded,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B*C, 1, H, W]
        
        output = result.reshape(B, C, H, W)
        
        return output


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn