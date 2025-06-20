import torch
import torch.nn as nn
import torch.nn.functional as F

class VitExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14', device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    def get_feature_from_input(self, input_img):
        """
        Returns patch token features (excluding CLS token).
        """
        output = self.forward(input_img)  # [B, N+1, D]
        return output[:, 1:]  # Exclude CLS token

    def get_cls_token(self, input_img):
        """
        Returns CLS token (useful for global feature).
        """
        output = self.forward(input_img)
        return output[:, 0]  # CLS token
