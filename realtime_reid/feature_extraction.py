import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from .resnet_base import ft_net, PCB, PCB_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Parameters
MODEL_PATH = 'checkpoints/net_last.pth'
N_CLASSES = 751
STRIDE = 2
LINEAR_NUM = 512


class ResNetReID():
    def __init__(
        self,
        use_PCB: bool = False,
        use_ibn: bool = False,
        custom_model_path: str = MODEL_PATH,
    ):

        self.use_PCB = use_PCB
        self.use_ibn = use_ibn

        # Init Data Transform Pipeline
        self.model = None
        self.init_data_transform()
        load_success = self.load_network(custom_model_path)
        if not load_success:
            raise RuntimeError("Model is not load successfully.")

        self.model = self.model.eval()
        self.model = self.model.to(device)

    def init_data_transform(self):
        if self.use_PCB:
            h, w = 384, 192
        else:
            h, w = 256, 128

        self.data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # def load_network(self, model_structure, pt_model_path: str = MODEL_PATH):
    def load_network(self, pt_model_path):
        """
        Load model checkpoint from the pth file, then return the loaded model.
        """

        load_success = True
        # Init model structure
        model_structure = ft_net(
            class_num=N_CLASSES,
            stride=STRIDE,
            ibn=self.use_ibn,
            linear_num=LINEAR_NUM,
        )

        if self.use_PCB:
            model_structure = PCB(N_CLASSES)

        try:
            print(f"Loading model from {pt_model_path}...")
            model_structure.load_state_dict(torch.load(pt_model_path))
            self.model = model_structure

            # Remove the final fc layer and classifier layer
            if self.use_PCB:
                self.model = PCB_test(self.model)
                print('finish loading model')
            else:
                self.model.classifier.classifier = nn.Sequential()
        except FileNotFoundError:
            print(
                f"Failed to load model from {pt_model_path},",
                "did you train the model?")
            load_success = False
        return load_success

    @staticmethod
    def fliplr(img: torch.Tensor):
        """flip horizontal"""
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
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
            - If input a string, it should be a path to an image.
            - If input a numpy array, it should be the image itself.

        Returns
        -------
        the feature (tensor) of the input image.
        """

        # Handle different input type
        img = None
        if isinstance(input_img, str):
            img = Image.open(input_img)
        elif isinstance(input_img, np.ndarray):
            img = Image.fromarray(input_img)
        else:
            raise TypeError(f"Unexpected img type, got {type(input_img)}")

        img = self.data_transforms(img)
        img = img.unsqueeze(0)
        n, c, h, w = img.size()

        ff = torch.FloatTensor(n, LINEAR_NUM).zero_()
        if self.use_PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()
        ff = ff.to(device)

        for i in range(2):
            if (i == 1):
                img = self.fliplr(img)
            input_img = img.to(device)

            with torch.no_grad():
                outputs = self.model(input_img)
            ff += outputs

        # norm feature
        if self.use_PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every
            # 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the
            # whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        return ff
