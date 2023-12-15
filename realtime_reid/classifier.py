import os
import torch

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonReID():
    def __init__(
        self,
        from_scratch: bool = False,
        from_file: str = None,
        from_tensor: torch.Tensor = None
    ):
        """
        Init the gallery from reading a torch.tensor saved in a `.pt` file or
        init a new torch.tensor. There are 3 options to init a gallery, which
        represented by the 3 parameters. These path SHOULDN'T be set add the
        same time.

        Parameters
        ----------
        from_scratch: bool, default False
            Whether to init the tensor from scratch (create an empty tensor).

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

        self.CONFIDENT_THRESHOLD = {
            'extreme': 0.95,
            'normal': 0.65,
        }
        self.gallery = self.init_gallery(from_scratch)
        self.ids = []
        self.current_max_id = 0

    def init_gallery(self, from_scratch: bool = False) -> torch.Tensor:
        """Init the gallery"""
        SAVED_PATH = os.path.join(MODULE_PATH, 'checkpoints/gallery.pt')

        if not from_scratch:
            if os.path.exists(SAVED_PATH):
                gallery = torch.load(SAVED_PATH, map_location=device)
            else:
                print("Can't find your path.")
                from_scratch = True

        if from_scratch:
            gallery = torch.Tensor()

        gallery = gallery.to(device)
        return gallery

    def calculate_score(self, target: torch.Tensor) -> torch.Tensor:
        """Calculate the relation score (Cosine) between target and items in
        the gallery."""
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        results = cos(target, self.gallery)
        return results

    def identify(
        self,
        target: torch.Tensor,
        update_gallery: bool = False
    ):
        """
        Get the ID of the input target.

        Parameters
        ----------
        target: torch.Tensor, required
            The (feature/embeddings) tensor of size [1, 512].

        update_gallery: bool, default False
            Whether to update the gallery (and ids).

        Returns
        -------
        int, the ID of the target tensor.
        """
        # The default id is current_max_id,
        # The only other option (below) is the id of the best match person.
        current_id = self.current_max_id

        if self.gallery.shape[0] == 0:
            # When no one is detected
            if update_gallery:
                self.gallery = target
        else:
            results = self.calculate_score(target)

            # Set up to get top_k
            results = torch.tensor(results)
            k = min(10, self.gallery.shape[0])

            top_k = torch.topk(results, k=k)
            top_scores, top_ppl = (top.tolist() for top in top_k)

            if top_scores[0] > self.CONFIDENT_THRESHOLD['normal']:
                current_id = top_ppl[0]

            if top_scores[0] < self.CONFIDENT_THRESHOLD['extreme']:
                if update_gallery:
                    self.gallery = torch.cat((self.gallery, target), dim=0)

        if update_gallery:
            if self.current_max_id == current_id:
                self.current_max_id += 1
            self.ids.append(current_id)

        return current_id
