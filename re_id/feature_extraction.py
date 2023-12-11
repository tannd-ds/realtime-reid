import os
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from tqdm import tqdm
from model import ft_net

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h, w = 256, 128
# Model Parameters
MODEL_PATH = 're_id/checkpoints/net_last.pth'
N_CLASSES = 751
STRIDE = 2
LINEAR_NUM = 512


class ResNetReID():
    def __init__(self):

        # Init Data Transform Pipeline
        self.data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Init model structure
        self.model_structure = ft_net(
            class_num=N_CLASSES, 
            stride=STRIDE, 
            ibn=False,
            linear_num=LINEAR_NUM,
        )

        # Now load model from checkpoint
        self.model = self.load_network(self.model_structure)
        self.model.classifier.classifier = nn.Sequential() # Remove the final fc layer and classifier layer

        self.model = self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()

    def load_network(self, network):
        """
        Load model checkpoint from the pth file, then return the loaded model.
        """
        save_path = os.path.join(MODEL_PATH)
        network.load_state_dict(torch.load(save_path))
        return network


    def fliplr(self, img):
        """flip horizontal"""
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature_single(self, img_path: str):
        """
        Extract feature from a trained `model`, with the input is an Image (or list of Images).
        """
        img = PIL.Image.open(img_path)
        img = self.data_transforms(img)
        img = img.unsqueeze(0)
        n, c, h, w = img.size()

        ff = torch.FloatTensor(n, LINEAR_NUM).zero_()
        if use_gpu:
            ff.to(device)

        for i in range(2):
            if(i==1):
                img = self.fliplr(img)
            input_img = img
            if (use_gpu):
                input_img = Variable(img.cuda()) 

            with torch.no_grad():
                outputs = self.model(input_img) 
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff

model = ResNetReID()

img_path = '../human_reidentification/runs/detect/exp/image0.jpg'
for i in range(20):
    test_feature = model.extract_feature_single(img_path)
    print(test_feature, test_feature.size())