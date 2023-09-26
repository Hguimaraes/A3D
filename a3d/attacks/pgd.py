import logging
import torch
import torch.nn as nn
from a3d.attacks import get_energy_ratio

class PGD(nn.Module):
    """This class implements the Projected Gradient Descent Attack 
    as described in the paper: [https://arxiv.org/abs/1706.06083]
    The implementation is based on
        https://github.com/Harry24k/adversarial-attacks-pytorch

    Arguments
    ---------
    model: nn.Module
        Torch model to be attacked
    loss: nn.Module
        Torch loss function (e.g. crossEntropyLoss) to compute the gradients
    alpha_snr:float (default: 40)
        Step size for distortion level in dB
    eps_snr:float (default: 20)
        Maximum distortion level in dB
    steps: int (default: 10)
        Number of iterations in the iterative FGSM
    random_start: bool (default: True)
        Whether to start from a random point or the original input

    Example
    -------
    >>> attacker = PGD(model, loss, eps_snr=40)
    >>> adversarial_sample = attacker(utterance, true_labels, target_labels)
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        alpha_snr: float = 40.,
        eps_snr: float = 20.,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        steps: int=10,
        random_start: bool = True,
        order: str = 'Linf'
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.alpha_snr = alpha_snr
        self.eps_snr = eps_snr
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.steps = steps
        self.random_start = random_start
        self.order = order

        if self.order not in ['Linf', 'L2']:
            raise ValueError("Order must be either Linf or L2")

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ):
        X = X.clone().detach().to(X.device)
        y = y.clone().detach().to(y.device)

        eps = get_energy_ratio(X, torch.ones_like(X), self.eps_snr)
        alpha = get_energy_ratio(X, torch.ones_like(X), self.alpha_snr)

        X_adv = X.clone().detach().to(X.device)
        if self.random_start:
            # @TODO: Implement random start without a for loop
            noise_start = []
            for idx, sample in enumerate(X_adv):
                # Starting at a uniformly random point
                noise_start.append(torch.empty_like(sample).uniform_(-eps[idx].item(), eps[idx].item()))
            delta = torch.stack(noise_start)

            if self.order == "L2":
                batch_size = X_adv.shape[0]
                delta_norm = delta.norm(p=2, dim=1).view(batch_size, 1)
                factor = torch.min(eps / delta_norm, torch.ones_like(delta_norm))
                delta = delta*factor

            X_adv = torch.clamp(X_adv + delta, min=self.clip_min, max=self.clip_max).detach()

        # PGD Iterations
        for _ in range(self.steps):
            X_adv.requires_grad = True
            outputs = self.model(X_adv)

            if self.targeted:
                cost = -self.loss_fn(outputs, y)
            else:
                cost = self.loss_fn(outputs, y)

            # Compute gradient w.r.t the inputs
            grad = torch.autograd.grad(cost, X_adv)[0]
            noise_adv = self.get_adversarial_noise(grad, alpha)

            # Adversarial perturbation
            if self.order == 'Linf':
                X_adv = X_adv.detach() + noise_adv
                delta = torch.clamp(X_adv - X, min=-eps, max=eps)

            else:
                # Normalize the gradient
                batch_size = X_adv.shape[0]
                grad_norms = torch.norm(grad, p=2, dim=1)
                grad = grad / grad_norms.view(batch_size, 1)

                # Compute the projected gradient with the L2 norm
                X_adv = X_adv.detach() + noise_adv
                delta = X_adv - X
                delta_norms = torch.norm(delta, p=2, dim=1)

                factor = eps / delta_norms.view(batch_size, 1)
                factor = torch.min(factor, torch.ones_like(factor))
                delta = delta * factor

            X_adv = torch.clamp(X + delta, min=self.clip_min, max=self.clip_max).detach()

        return X_adv, delta

    def get_adversarial_noise(self, grads, alpha):
        grads = grads.clone().detach().to(grads.device).sign()
        grads = grads.squeeze(0)

        return alpha*grads
