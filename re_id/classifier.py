import os
import torch

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonReID():
    def __init__(self, from_scratch: bool = False, custom_tensor_path: str = None):
        """Init a Classifier"""
        self.gallery = self.init_gallery(from_scratch, custom_tensor_path)

    def init_gallery(
        self,
        from_scratch: bool = False,
        from_file: str = None,
        from_tensor: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Init the gallery from reading a torch.tensor saved in a `.pt` file or init a new torch.tensor.
        There are 3 options to init a gallery, which represented by the 3 parameters.
        These path SHOULDN'T be set add the same time.

        Parameters
        ----------
        from_scratch: bool, default False
            Wheater to to init the tensor from scratch (create an empty tensor).

        from_file: str, default None
            A path, a custom saved tensor saved on the disk.
            If this is set, load the saved tensor in path to be gallery.

        from_tensor: torch.Tensor, default None
            A tensor. It this is set, load the given tensor to be gallery.

        Returns
        -------
        torch.Tensor

        TODOs
        -----
        Handle `custom_path`
        """
        SAVED_PATH = os.path.join(MODULE_PATH, 'checkpoints/gallery.pt')

        if os.path.exists(SAVED_PATH):
            gallery = torch.load(SAVED_PATH, map_location=device)
        else:
            print("Can't find your path.")
            from_scratch = True

        if from_scratch:
            gallery = torch.zeros(1, 512)

        return gallery

    def calculate_score(self, target: torch.Tensor):
        """
        Calculate the relation score (Cosine) between target and items in gallery.
        """
        THRESHOLD = 0.5

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        results = cos(target, self.gallery)

        if results.max() < THRESHOLD:
            self.gallery = torch.cat((self.gallery, target), dim=0)
        print(results, torch.argmax(results, dim=-1))
