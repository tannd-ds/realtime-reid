import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from .resnet_base import FtNet, FtNetDense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h, w = 256, 128
# ResNet50 Parameters
# Train: --train_all
MODEL_PATH = 'checkpoints/net_last.pth'
N_CLASSES = 751
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


class PersonDescriptor():
    def __init__(self, use_dense=False):

        # Init Data Transform Pipeline
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((h, w), interpolation=3, antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Init model structure
        if use_dense:
            self.model_structure = FtNetDense(
                class_num=N_CLASSES,
                droprate=DENSE_DROPRATE,
                stride=DENSE_STRIDE,
                circle=DENSE_CIRCLE,
                linear_num=LINEAR_NUM,
            )
            self.model = self.load_network(
                self.model_structure,
                DENSE_MODEL_PATH
            )
        else:
            self.model_structure = FtNet(
                class_num=N_CLASSES,
                stride=STRIDE,
                linear_num=LINEAR_NUM,
            )
            self.model = self.load_network(
                self.model_structure,
                MODEL_PATH
            )

        # Remove the final fc layer and classifier layer
        self.model.classifier.classifier = nn.Sequential()

        self.model = self.model.eval()
        self.model = self.model.to(device)

    def load_network(self, model_structure, pt_model_path: str = MODEL_PATH):
        """
        Load model checkpoint from the pth file, then return the loaded model.
        """
        try:
            print(f"Loading model from {pt_model_path}...")
            model_structure.load_state_dict(torch.load(pt_model_path))
        except FileNotFoundError:
            print(
                f"Failed to load model from {pt_model_path},",
                "did you train the model?")
        finally:
            print("Model loaded successfully!")
        return model_structure

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
        n, c, h, w = img.size()

        feature_map = torch.FloatTensor(n, LINEAR_NUM).zero_()
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
        fnorm = torch.norm(feature_map, p=2, dim=1, keepdim=True)
        feature_map = feature_map.div(fnorm.expand_as(feature_map))

        return feature_map
