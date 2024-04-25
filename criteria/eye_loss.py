import torch
from torch import nn

from metrics.FAN import FAN


class EyeLoss(nn.Module):
    def __init__(self):
        super(EyeLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.model = FAN(4, "False", "False", 98)
        self.setup_model('./pretrained_models/WFLW_4HG.pth')

    def setup_model(self, path_to_model: str):
        checkpoint = torch.load(path_to_model)
        if 'state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model.load_state_dict(model_weights)
        self.model.eval()

    def detect_landmarks(self, input):
        mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(input.device)
        std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(input.device)
        input = (std * input) + mean

        output, _ = self.model(input)
        pred_heatmap = output[-1][:, :-1, :, :]
        return pred_heatmap[:, 96, :, :], pred_heatmap[:, 97, :, :]

    def forward(self, y_hat, y):
        y_hat_heatmap_left, y_hat_heatmap_right = self.detect_landmarks(y_hat)
        y_heatmap_left, y_heatmap_right = self.detect_landmarks(y)
        return self.l2_loss(y_hat_heatmap_left, y_heatmap_left) + self.l2_loss(y_hat_heatmap_right, y_heatmap_right)
