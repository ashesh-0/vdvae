"""
Here, we ensure that contrastive loss
"""
import sys

import torch
import torch.nn as nn

sys.path.append("/home/ashesh/ashesh/PyTorch_CIFAR10/")
from cifar10_models.densenet import densenet121


class DisentanglementModule(nn.Module):
    def __init__(self, latent_size_dict) -> None:
        super().__init__()
        self._lsizes = latent_size_dict
        self._image_model = densenet121(pretrained=True, device='cpu')
        self._image_model.classifier = nn.Identity()
        for param in self._image_model.parameters():
            param.requires_grad = False

        self._tau_pos = 0.1
        self._tau_neg = 0.1
        self._w = 0.5

    def _contrastive_loss_single(self, rep1, rep2, same_level):
        dis = nn.MSELoss()
        if same_level:
            return torch.max(dis(rep1, rep2) - self._tau_pos, 0)[0]
        else:
            return torch.max(self._tau_neg - dis(rep1, rep2), 0)[0]

    def _compute_C_pos_neg(self, noise_levels):
        batch_size = len(noise_levels)
        c_pos = 0
        c_neg = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                same_level = int(noise_levels[i] == noise_levels[j])
                c_pos += same_level
                c_neg += 1 - same_level
        return c_pos, c_neg

    def _get_contrastive_loss(self, representation, noise_levels):
        batch_size = len(noise_levels)
        loss = 0
        c_pos, c_neg = self._compute_C_pos_neg(noise_levels)
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                same_level = noise_levels[i] == noise_levels[j]
                if same_level:
                    loss_elem = 1 / c_pos * self._contrastive_loss_single(representation[i], representation[j],
                                                                          same_level)
                else:
                    loss_elem = 1 / c_neg * self._contrastive_loss_single(representation[i], representation[j],
                                                                          same_level)
                loss += loss_elem
        return loss

    def get_loss(self, activations, pred_image, noise_levels):
        latent_loss = 0
        representation = self._image_model(pred_image.permute(0, 3, 1, 2))
        output_contrastive_loss = self._get_contrastive_loss(representation, noise_levels)
        cnt = 0
        for key in activations:
            if self._lsizes[key] == 0:
                continue
            N = self._lsizes[key]
            cnt += 1
            noise_z = activations[key][:, :N]
            latent_loss += self._get_contrastive_loss(noise_z, noise_levels)
        latent_loss = latent_loss / cnt
        return self._w * latent_loss + (1 - self._w) * output_contrastive_loss


if __name__ == '__main__':
    import numpy as np
    N = 16
    mod = DisentanglementModule({1: 0, 2: 0, 4: 0, 8: 0, 16: 50, 32: 50})
    activations = {32: torch.Tensor(np.random.rand(N, 200, 32, 32))}
    print(
        mod.get_loss(activations, torch.Tensor(np.random.rand(N, 32, 32, 3)), torch.Tensor(np.random.choice(8,
                                                                                                            size=N))))
