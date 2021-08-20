"""
Here, we ensure that contrastive loss
"""
import torch
import torch.nn as nn
from cifar10_models.densenet import densenet121


class DisentanglementModule:
    def __init__(self, latent_size_dict) -> None:
        self._lsizes = latent_size_dict
        self._image_model = densenet121(pretrained=True, device='cpu')
        self._image_model.classifier = nn.Identity()
        self._tau_pos = 0.1
        self._tau_neg = 0.1

    def _contrastive_loss_single(self, rep1, rep2, same_level):
        dis = nn.MSELoss(dim=0)
        if same_level:
            return torch.max(dis(rep1, rep2) - self._tau_pos, 0)
        else:
            return torch.max(self._tau_neg - dis(rep1, rep2), 0)

    def _get_contrastive_loss(self, representation, noise_levels):
        batch_size = len(noise_levels)
        loss = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                same_level = noise_levels[i] == noise_levels[j]
                loss += self._contrastive_loss_single(representation[i], representation[j], same_level)
        return loss / (0.5 * batch_size * (batch_size - 1))

    def get_loss(self, activations, dec_output, noise_levels):
        loss_dict = {}
        for key in activations:
            if self._lsizes[key] == 0:
                continue
            N = self._lsizes[key]

            noise_z = activations[:, :N]
            latent_loss = self._get_contrastive_loss(noise_z, noise_levels)

            representation = self._image_model(dec_output)
            output_contrastive_loss = self._get_contrastive_loss(representation, noise_levels)

            loss_dict[key] = self._w * latent_loss + (1 - self._w) * output_contrastive_loss
        return loss_dict
