import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from learning.utils.model import enable_dropout
from torch.nn import SmoothL1Loss
import torch.nn.init as init
import numpy as np
from torchvision.models.resnet import conv1x1, conv3x3

def _get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2
    return padding

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes,stride=1):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 =  nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes) 
        self.stride = stride

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        conv2 = self.conv2(x)
        conv2 = self.bn2(conv2)

        return conv1 + conv2

class MonteCarloDronet(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloDronet, self).__init__()
        print('Loading DroNet')
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p = kwargs.get('p', 0.1)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 1)
        self.input_shape = kwargs.get('input_shape', (60, 80))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32,kernel_size=5,stride=2),
            nn.MaxPool2d(kernel_size=(3,3), stride=[2,2]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32 ,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32,64,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64,128 ,stride=2),#TODO check conv regularizer
            Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),# use self.p in the future
        )

        self.num_feats_extracted = 2560
        self.omega_channel = nn.Sequential(
            nn.Linear(self.num_feats_extracted,1)
        )

        # Corner detected channel
        self.corner =nn.Sequential(
            nn.Linear(self.num_feats_extracted, 1)
        )
        
        # Collision detected channel
        self.collision =nn.Sequential(
            nn.Linear(self.num_feats_extracted, 1)
        )

        self.max_speed = torch.tensor(0.75).to(self._device)
        self.min_speed = torch.tensor(0.35).to(self._device)
        self.stop_speed_threshold = torch.tensor(0.12).to(self._device)
        self.stop_speed = torch.tensor(0,dtype=torch.float).to(self._device)
        self.mask_zero = torch.tensor(0).to(self._device)
        self.mask_one = torch.tensor(1).to(self._device)
        self.speed_threshold = 0.5
        self.decay = 1/10
        self.alpha = 0
        self.epoch_0 = 10
        self.epoch = 0

    def forward(self, images):
        features = self.feature_extractor(images)
        omega = self.omega_channel(features)
        corner_detect = self.corner(features)
        collision_detect = self.collision(features)
        return corner_detect, collision_detect, omega
    

    def loss(self, *args):
        self.train()
        images, target = args
        corner_detect, collision_detected, omega = self.forward(images) 
        criterion_bce = nn.BCEWithLogitsLoss()
        is_slowing_down = (target[:,0]> self.speed_threshold).float().unsqueeze(1)  # 1 for expert speeding up and 0 for slowing down for a corner or an incoming duckbot
        is_colliding = (target[:,0]<self.stop_speed_threshold).float().unsqueeze(1) # 1 for collision detected and 0 for no collision
        loss_omega = F.mse_loss(omega, target[:,1].unsqueeze(1), reduction='mean')
        loss_corner = criterion_bce(corner_detect, is_slowing_down)
        loss_coll = criterion_bce(collision_detected, is_colliding )
        loss = loss_omega + (loss_corner + loss_coll) * max(0, 1 - np.exp(self.decay * (self.epoch - self.epoch_0)))
        return loss
    

    def predict(self, *args):
        images = args[0]
        prob_corner, prob_coll, omega = self.forward(images)
        # post processing v values to its max and min counterparts

        prob_coll = torch.sigmoid(prob_coll)
        prob_corner = torch.sigmoid(prob_corner)
        coll_mask = torch.where(prob_coll>0.5 , self.mask_zero, self.mask_one )
        v_tensor = (1 - prob_corner) * self.max_speed + (prob_corner * self.min_speed)
        v_tensor[coll_mask==0] = self.stop_speed 
        omega[coll_mask==0] = self.stop_speed
        output = torch.cat((v_tensor , omega ), 1)
        return output

    def predict_with_uncertainty(self, *args):
        # Set model to evaluation except dropout layers
        self.eval()
        enable_dropout(self)

        # Sample multiple times from the ensemble of models
        prediction = []
        for i in range(self.num_samples):
            prediction.append(self.predict(*args))

        # Calculate statistics of the outputs
        prediction = torch.stack(prediction)
        mean = prediction.mean(0)
        var = prediction.var(0)

        return mean.squeeze().tolist(), var.squeeze().tolist()

if __name__ == '__main__':
    #TODO test the model input and output
    pass