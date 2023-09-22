import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CW(nn.Module):
    """
    This class implements of the CW algorithm as in the paper [https://arxiv.org/abs/1608.04644]
    Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        c: float = 40.,
        tau: float = 20.,
        lr: float = 1e-2,
        steps: int=10,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.c = c
        self.tau = tau
        self.lr = lr
        self.steps = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ):
        X = X.clone().detach().to(X.device)
        y = y.clone().detach().to(y.device)

        w = torch.atanh(X) # torch.zeros_like(X).detach()
        w.requires_grad = True

        best_X_adv = X.clone().detach()
        best_norm_loss = 1e10*torch.ones((best_X_adv.size(0))).to(X.device)

        dim = len(X.shape)

        optimizer = optim.Adam([w], lr=self.lr)
        for step in range(self.steps):
            # Get adversarial samples in the tanh space
            X_adv = torch.tanh(w)

            # Delta norm
            current_loss = F.mse_loss(X_adv, X, reduction='none').sum(dim=1)
            loss = current_loss.sum()

            # Output of the model
            outputs = self.model(X_adv)
            f_loss = self.loss_fn(outputs, y).sum()

            cost = loss - self.c*f_loss
            # print(f"step={step}", loss, -self.c*f_loss, cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial sample
            pred_labels = torch.argmax(outputs, dim=1)
            correct = (pred_labels == y).float()

            # Filter out samples that do not decrease the loss
            mask = (1-correct)*(best_norm_loss > current_loss.detach())
            mask_snr = self.snr_loss(X_adv, X) > self.tau
            mask = mask*mask_snr

            best_norm_loss = mask*current_loss.detach() + (1-mask)*best_norm_loss
            mask = mask.view([-1, dim-1])
            best_X_adv = mask*X_adv.detach() + (1-mask)*best_X_adv

        # noise = best_X_adv - X
        # S_signal = (X*X).sum(1) / X.size(1) # Batched dot-product
        # S_noise = (noise*noise).sum(1) / noise.size(1)  # Batched dot-product
        # print(S_noise)
        # print(10*torch.log10(S_signal/(S_noise + 1e-8)).mean())
        return best_X_adv, best_X_adv - X

    def snr_loss(self, X_adv, X):
        x_psd = torch.max(X, dim=1)[0]
        delta_psd = torch.max(X_adv-X, dim=1)[0]

        return 20*torch.log10(x_psd/delta_psd)