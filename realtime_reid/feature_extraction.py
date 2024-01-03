import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from .resnet_base import ft_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h, w = 256, 128
# Model Parameters
MODEL_PATH = 'checkpoints/net_last.pth'
N_CLASSES = 751
STRIDE = 2
LINEAR_NUM = 512


class PersonDescriptor():
    def __init__(self):

        # Init Data Transform Pipeline
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((h, w), interpolation=3),
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
            feature_map += outputs

        # Normalize feature
        fnorm = torch.norm(feature_map, p=2, dim=1, keepdim=True)
        feature_map = feature_map.div(fnorm.expand_as(feature_map))

        return feature_map
