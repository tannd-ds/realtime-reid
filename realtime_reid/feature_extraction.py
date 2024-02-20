import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from .resnet_base import FtNet, FtNetDense, PCB, PCB_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h, w = 256, 128
# ResNet50 Parameters
# Train: --train_all
MODEL_PATH = 'checkpoints/net_last.pth'
STRIDE = 2
LINEAR_NUM = 512

# DenseNet121 Parameters
# Train: --warm_epoch 5 --stride 1 --erasing_p 0.5 \
#        --batchsize 8 --lr 0.02 --name dense_warm5_s1_b8_lr2_p0.5_circle \
#        --circle --use_dense
DENSE_MODEL_PATH = 'checkpoints/dense_net_last.pth'
DENSE_DROPRATE = 0.5
DENSE_STRIDE = 2  # Use stride 2 instead of 1 results in less id switch
DENSE_CIRCLE = True

# PCB Parameters
# Train:
PCB_MODEL_PATH = 'checkpoints/pcb_net_30.pth'


class PersonDescriptor:
    def __init__(self,
                 use_dense=False,
                 use_pcb=False,
                 model_path=None,
                 n_classes=751):

        self.use_dense = use_dense
        self.use_pcb = use_pcb
        self.n_classes = n_classes

        if self.use_pcb:
            self.w, self.h = 384, 192
        else:
            self.w, self.h = 256, 128

        # Init Data Transform Pipeline
        self.data_transforms = self.init_data_transforms()

        # Init Model
        self.model_structure, self.model = self.init_model(model_path)

        # Remove the final fc layer and classifier layer
        if self.use_pcb:
            self.model = PCB_test(self.model)
        else:
            self.model.classifier.classifier = nn.Sequential()

        self.model = self.model.eval()
        self.model = self.model.to(device)

    def init_model(self, model_path: str = None):
        """Init model structure"""
        if self.use_pcb:
            model_structure = PCB(class_num=self.n_classes)
            model = self.load_network(
                model_structure,
                PCB_MODEL_PATH if model_path is None else model_path
            )
        elif self.use_dense:
            model_structure = FtNetDense(
                class_num=self.n_classes,
                droprate=DENSE_DROPRATE,
                stride=DENSE_STRIDE,
                circle=DENSE_CIRCLE,
                linear_num=LINEAR_NUM,
            )
            model = self.load_network(
                model_structure,
                DENSE_MODEL_PATH if model_path is None else model_path
            )
        else:
            model_structure = FtNet(
                class_num=self.n_classes,
                stride=STRIDE,
                linear_num=LINEAR_NUM,
            )
            model = self.load_network(
                model_structure,
                MODEL_PATH if model_path is None else model_path
            )

        return model_structure, model

    @staticmethod
    def load_network(model_structure, pt_model_path: str = MODEL_PATH):
        """
        Load model checkpoint from the pth file, then return the loaded model.
        """
        try:
            print(f"Loading model from {pt_model_path}...")
            model_structure.load_state_dict(torch.load(pt_model_path))
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(
                f"Failed to load model from {pt_model_path},",
                "did you train the model?")
        return model_structure

    def init_data_transforms(self):
        """Initialize the data transform pipeline."""
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.h, self.w), interpolation=3, antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return data_transforms

    @staticmethod
    def fliplr(img: torch.Tensor):
        """flip horizontal"""
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
        inv_idx = inv_idx.to(device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(
        self,
        input_img: str | np.ndarray
    ) -> torch.Tensor:
        """
        Extract feature of an Image.

        Parameters
        ----------
        input_img: str | np.ndarray
            The image that need to be extract feature.
            - If input a string, it should be a path to an image
            - If input a numpy array, it should be the image itself.

        Returns
        -------
        the feature (tensor) of the input image.
        """

        # Handle different input type
        img = None
        if isinstance(input_img, str):
            img = cv2.imread(input_img)
        elif isinstance(input_img, np.ndarray):
            img = input_img
        else:
            raise TypeError(f"Unexpected img type, got {type(input_img)}")

        img = self.data_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        n, c, _, _ = img.size()

        feature_map = torch.FloatTensor(n, LINEAR_NUM).zero_()
        if self.use_pcb:
            # We have 6 parts -> 6 feature maps
            feature_map = torch.FloatTensor(n, 2048, 6).zero_()
        feature_map = feature_map.to(device)

        for i in range(2):
            if i == 1:
                img = self.fliplr(img)

            with torch.no_grad():
                outputs = self.model(img)

            if isinstance(outputs, list):
                outputs = outputs[0]
            feature_map += outputs

        # Normalize feature
        if self.use_pcb:
            f_norm = torch.norm(feature_map, p=2, dim=1, keepdim=True) * np.sqrt(6)
            feature_map = feature_map.div(f_norm.expand_as(feature_map))
            feature_map = feature_map.view(feature_map.size(0), -1)
        else:
            f_norm = torch.norm(feature_map, p=2, dim=1, keepdim=True)
            feature_map = feature_map.div(f_norm.expand_as(feature_map))

        return feature_map
