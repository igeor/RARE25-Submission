import torch
import timm
from torch import nn
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List, Optional

class ClassificationHead(nn.Module):
    """Small supervised head used to monitor Domain Adaptation """

    def __init__(self, in_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_classification_head(
        in_dim: int, 
        hidden_dims: List[int], 
        num_classes: int = 1,
        hidden_activation: str = "relu",   # Fix: Use non-linearities for hidden layers
        norm: Optional[str] = None,        # Options: "batch", "layer", None
        dropout: float = 0.1
) -> nn.Module:
    
    # If no hidden layers, return your original class directly
    if not hidden_dims: 
        return ClassificationHead(in_dim, num_classes)
    
    # Define hidden activation factory
    if hidden_activation == "relu": act_fn = nn.ReLU
    elif hidden_activation == "gelu": act_fn = nn.GELU
    else: act_fn = nn.Identity

    # Define normalization factory
    if norm == "batch": norm_fn = nn.BatchNorm1d
    elif norm == "layer": norm_fn = nn.LayerNorm
    else: norm_fn = None
        
    # Build the intermediate hidden MLP structure
    layers = []
    current_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if norm_fn is not None:
            layers.append(norm_fn(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    # Initialize your original class using the last hidden dimension
    final_head = ClassificationHead(current_dim, num_classes)
    layers.append(final_head)
    
    # Wrap the entire pipeline cleanly into a single module
    return nn.Sequential(*layers)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        model_name,
        checkpoint_path_list,
        image_size=224,
        head_hidden_dims=None,
        head_activation="relu",
        head_norm=None,
        head_dropout=0.1,
        device=None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not checkpoint_path_list:
            raise ValueError("checkpoint_path_list must contain at least one checkpoint.")

        self.models = nn.ModuleList()

        for checkpoint_path in checkpoint_path_list:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=1,
            )

            model.fc = get_classification_head(
                in_dim=model.num_features,
                hidden_dims=head_hidden_dims or [],
                hidden_activation=head_activation,
                norm=head_norm,
                dropout=head_dropout,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.models.append(model)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, x):
        probs = []

        for model in self.models:
            logits = model(x)
            prob = torch.sigmoid(logits)
            probs.append(prob)

        return torch.stack(probs, dim=0).mean(dim=0)

    def predict(self, images: list[np.ndarray]):
        """
        Accepts a list of numpy images (HWC, uint8 or float),
        converts them to PIL Images, applies transforms, and runs inference.
        """
        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        probs = []
        for img in pil_images:
            img = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prob = self.forward(img).squeeze().cpu().item()

            probs.append(prob)

        return probs