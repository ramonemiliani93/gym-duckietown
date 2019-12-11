import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from learning.utils.model import enable_dropout
from torch.nn import SmoothL1Loss
import torch.nn.init as init
import numpy as np
from torchvision.models.resnet import conv1x1, conv3x3




def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / (np.array(effective_num) + 1e-6)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels.to(torch.int64), no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot 
    weights = weights.squeeze().sum(1).sum(1) 
    # weights = weights.unsqueeze(1)
    # weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        criterion = nn.BCEWithLogitsLoss(weight = weights.unsqueeze(1))
        cb_loss = criterion(logits, labels)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

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
        self.steering_angle_channel = nn.Sequential(
            nn.Linear(self.num_feats_extracted,1)
        )

        # predicting should i speed up or not
        self.speed_up_channel =nn.Sequential(
            nn.Linear(self.num_feats_extracted, 1)
        )

        # Collision detection channel
        self.collision =nn.Sequential(
            nn.Linear(self.num_feats_extracted, 1)
        )

        self.decay = 1/10
        self.alpha = 0
        self.epoch_0 = 10
        self.epoch = 0
        self.mask_zero = torch.tensor(0).to(self._device)
        self.mask_one = torch.tensor(1).to(self._device)
        self.set_max_velocity()
    
    def set_max_velocity(self, max_velocity = 0.75):
        self.max_velocity = max_velocity
        self.max_speed_tensor = torch.tensor(self.max_velocity).to(self._device)
        self.min_speed_pure_pursuit = (self.max_velocity) * 0.5
        self.stop_speed_threshold = torch.tensor(0.14).to(self._device)
        self.stop_speed = torch.tensor(0,dtype=torch.float).to(self._device)

    def forward(self, images):
        features = self.feature_extractor(images)
        steering_angle = self.steering_angle_channel(features)
        is_speed_up = self.speed_up_channel(features)
        collision_detect = self.collision(features)
        return is_speed_up, collision_detect, steering_angle

    def loss(self, *args):
        self.train()
        images, target = args
        is_speed_up, collision_detect, steering_angle = self.forward(images) 
        
        is_colliding = (target[:,0]<self.stop_speed_threshold).float().unsqueeze(1) 
        speed_up = (target[:,0] > self.min_speed_pure_pursuit).float().unsqueeze(1)  # 0 for expert speeding up and 1 for slowing down for a corner or an incoming duckbot
        loss_steering_angle = F.mse_loss(steering_angle, target[:,1].unsqueeze(1), reduction='mean')
        samples_per_cls = [torch.where(speed_up==0)[0].shape[0] , torch.where(speed_up==1)[0].shape[0]]
        # loss_v = criterion(is_speed_up, speed_up)
        loss_v = CB_loss(speed_up, is_speed_up,samples_per_cls,2,'sigmoid',0.999,2.0)
        # loss_obstacle = criterion(collision_detect, is_colliding)
        samples_per_cls = [torch.where(is_colliding==0)[0].shape[0] , torch.where(is_colliding==1)[0].shape[0]]
        loss_obstacle = CB_loss(is_colliding, collision_detect,samples_per_cls,2,'sigmoid',0.999,2.0)
        loss = loss_steering_angle + ( loss_v * max(0, 1 - np.exp(self.decay * (self.epoch - self.epoch_0))) )+ ( loss_obstacle * max(0, 1 - np.exp(self.decay * (self.epoch - self.epoch_0*1.5))) )
        return loss
    

    def predict(self, *args):
        images = args[0]
        is_speed_up, prob_coll, steering_angle = self.forward(images)
        prob_coll = torch.sigmoid(prob_coll)
        is_speed_up = torch.sigmoid(is_speed_up)
        coll_mask = torch.where(prob_coll>0.5 , self.mask_zero, self.mask_one )[0]
        v_tensor  =  (is_speed_up) * self.max_speed_tensor + (1 - is_speed_up) * self.min_speed_pure_pursuit  # torch.where(prob_corner>0.5, self.min_speed_tensor, self.max_speed_tensor )  
        steering_angle =  steering_angle * np.pi / 2

        v_tensor[coll_mask==0] = self.stop_speed 
        steering_angle[coll_mask==0] = self.stop_speed
        output = torch.cat((v_tensor, steering_angle), 1)
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