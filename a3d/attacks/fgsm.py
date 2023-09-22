import torch
import torch.nn as nn
from a3d.sigproc import get_energy_ratio

class FGSM(nn.Module):
    """This class implements the Fast-Gradient Sign Method
    as described in the paper:
    The implementation is based on https://github.com/Harry24k/adversarial-attacks-pytorch

    Arguments
    ---------
    model: nn.Module
        Torch model to be attacked
    loss: nn.Module
        Torch loss function (e.g. crossEntropyLoss) to compute the gradients
    eps_snr:float (default: 40)
        Distortion level in dB

    Example
    -------
    >>> attacker = FGSM(model, loss, eps_snr=40)
    >>> adversarial_sample = attacker(utterance, true_labels, target_labels)
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eps_snr: float = 40.,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        targeted: bool = False
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps_snr = eps_snr
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ):
        X = X.clone().detach().to(X.device) # torch.Size([B, L])
        y = y.clone().detach().to(y.device) # torch.Size([B, C]) or torch.Size([B])

        X.requires_grad = True
        outputs = self.model(X)

        if self.targeted:
            cost = -self.loss_fn(outputs, y)
        else:
            cost = self.loss_fn(outputs, y)

        # Compute gradient w.r.t the inputs
        grad = torch.autograd.grad(
            cost, X, retain_graph=False, create_graph=False
        )[0]

        # FGSM
        _, noise_adv = self.get_adversarial_noise(X, grad, X.device)
        X_adv = X + noise_adv
        X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max).detach()

        return X_adv, noise_adv

    def get_adversarial_noise(self, audio, grads, device):
        grads = grads.clone().detach().to(device).sign()
        # grads = grads.squeeze(0)

        audio = audio.clone().detach().to(device)
        # audio = audio.squeeze(0)
        eps = get_energy_ratio(audio, grads, self.eps_snr)

        return eps, eps*grads