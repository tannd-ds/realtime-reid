import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonReID():
    def __init__(
        self,
        from_file: str = None,
        from_tensor: torch.Tensor = None
    ):
        """
        Init the embeddings from reading a torch.tensor saved in a `.pt` file
        or init a new torch.tensor. There are 3 options to init an embeddings,
        which represented by the 2 parameters. These paras SHOULDN'T be set at
        the same time.
        Default to init a new embeddings.

        Parameters
        ----------
        from_file: str, default None
            A path, a custom saved tensor saved on the disk.
            If this is set, load the saved tensor in path to be embeddings.

        from_tensor: torch.Tensor, default None
            A tensor. It this is set, load the given tensor to be embeddings.

        Returns
        -------
        torch.Tensor
        """
        assert from_file is None or from_tensor is None, \
            "from_file and from_tensor can't be set at the same time."

        # Set up the threshold
        self.CONFIDENT_THRESHOLD = {
            'extreme': 0.90,
            'normal': 0.85,
        }
        self.cos_scorer = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # Init the embeddings
        if from_file is not None:
            self.embeddings = torch.load(from_file, map_location=device)
        elif from_tensor is not None:
            self.embeddings = from_tensor
        else:
            self.embeddings = torch.Tensor()
        self.embeddings = self.embeddings.to(device)

        # Init the ids
        self.ids = []
        self.current_max_id = 0

    def calculate_score(
        self, target: torch.Tensor,
        score: str = 'cosine'
    ) -> torch.Tensor:
        """Calculate the relation score (Cosine) between
        the target and items in the embeddings."""
        results = None
        if score == "cosine":
            # Calculate score using Cosine Similarity
            results = self.cos_scorer(target, self.embeddings)
        return results

    def identify(
        self,
        target: torch.Tensor,
        update_embeddings: bool = False
    ):
        """
        Get the ID of the input target.

        Parameters
        ----------
        target: torch.Tensor, required
            The (feature/embeddings) tensor of size [1, 512].

        update_embeddings: bool, default False
            Whether to update the embeddings (and ids).

        Returns
        -------
        int, the ID of the target tensor.
        """
        # The default id is current_max_id,
        # The only other option (below) is the id of the best match person.
        target_id = self.current_max_id
        new_embeddings = self.embeddings

        if self.embeddings.shape[0] == 0:
            # When no one is detected
            new_embeddings = target
        else:
            results = self.calculate_score(target)

            # Get the best match person
            top_score, top_ppl = torch.max(results, dim=0)
            top_score, top_ppl = top_score.item(), top_ppl.item()
            if top_score > self.CONFIDENT_THRESHOLD['normal']:
                target_id = self.ids[top_ppl]

            new_embeddings = torch.cat((self.embeddings, target), dim=0)

        if update_embeddings:
            self.embeddings = new_embeddings
            if new_embeddings.shape[0] > len(self.ids):
                self.ids.append(target_id)
            if self.current_max_id == target_id:
                self.current_max_id += 1

        return target_id
