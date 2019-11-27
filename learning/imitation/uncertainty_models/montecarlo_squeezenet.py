import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from learning.utils.model import enable_dropout
from torch.nn import SmoothL1Loss
import torch.nn.init as init


class MonteCarloSqueezenet(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloSqueezenet, self).__init__()
        print('Loading Squeeze net')
        self.p = kwargs.get('p', 0.05)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 1)
        
        #TODO add squeezenet transformation to be used on observation images
        #TODO check finetuning parameters
        self.model = models.squeezenet1_0(pretrained=True)
        #TODO add dropout which is in classifier[0] nn.Dropout(p=0.5) with p value coming to the model
        final_conv = nn.Conv2d(512, self.num_outputs, kernel_size=(1,1), stride=(1,1))
        self.model.classifier[1] = final_conv
        self.model.num_classes = self.num_outputs
        self.episode = 0
        self.n_epochs = 0
        self.freeze_pretrained_modules()
        #self.criterion = SmoothL1Loss(reduction='mean')

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        #TODO add auto tune paramter for the first episode

    def forward(self, images):
        output = self.model(images)
        return output
    
    def freeze_pretrained_modules(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # unfreezing the classifier part
        for module in self.model.classifier:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def loss(self, *args):
        self.train()
        '''
        if self.episode==0:
            #TODO tune the warm start parameters
            if self.n_epochs >15:
                self.unfreeze_pretrained()
            self.n_epochs +=1
        '''
        images, target = args
        prediction = self.forward(images)
        #TODO update loss function to give more priority to omega predictions
        loss_velocity = F.mse_loss(prediction[:,0],target[:,0], reduction='mean')
        loss_omega = F.mse_loss(prediction[:,1],target[:,1], reduction='mean')
        loss = loss_velocity + 1.5 * loss_omega
        return loss

    def predict(self, *args):
        images = args[0]
        output = self.model(images)
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