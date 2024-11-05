"""
Loss functions for the CSAI model and baselines.

This module implements various loss functions used for training time series imputation models:
- SVAELoss: Loss function for Stochastic Variational Auto-Encoder
- DiceBCELoss: Combined Dice and Binary Cross Entropy loss
- VRNNLoss: Loss function for Variational RNN
- FocalLoss: Focal loss for handling class imbalance
- AsymSimilarityLoss: Asymmetric similarity loss
"""

from typing import Tuple, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class SVAELoss(nn.Module):
    """
    Loss function for Stochastic Variational Auto-Encoder component.
    Combines reconstruction loss with KL divergence regularization.
    """
    
    def __init__(self, args):
        """
        Args:
            args: Configuration object containing:
                - lambda1: L1 regularization weight
                - beta: KL divergence weight
                - device: Computing device
        """
        super().__init__()
        self.args = args
        self.lambda1 = torch.tensor(args.lambda1)
        self.mae = nn.L1Loss()

    def forward(
        self, 
        model: nn.Module,
        x: torch.Tensor,
        eval_x: torch.Tensor,
        x_bar: torch.Tensor,
        m: torch.Tensor,
        eval_m: torch.Tensor,
        enc_mu: torch.Tensor,
        enc_logvar: torch.Tensor,
        dec_mu: torch.Tensor,
        dec_logvar: torch.Tensor,
        phase: str = 'train'
    ) -> Tuple[torch.Tensor, float, float, float, float]:
        """
        Compute SVAE loss.

        Args:
            model: Neural network model
            x: Input tensor
            eval_x: Evaluation tensor
            x_bar: Reconstructed tensor
            m: Missing value mask
            eval_m: Evaluation mask
            enc_mu: Encoder mean
            enc_logvar: Encoder log variance
            dec_mu: Decoder mean
            dec_logvar: Decoder log variance
            phase: Training phase ('train' or 'eval')

        Returns:
            Tuple containing:
            - Total loss
            - Negative log likelihood
            - MAE
            - KL divergence
            - L1 regularization
        """
        # Reconstruction Loss (Negative Log Likelihood)
        nll = -Normal(dec_mu, torch.exp(0.5 * dec_logvar)).log_prob(x).sum(1)
        mae = torch.tensor([0.0]).to(self.args.device)
        recon_loss = nll

        # KL Divergence Loss
        kld = -0.5 * self.args.beta * torch.sum(
            1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp(), 1
        )

        # L1 Regularization
        l1_reg = torch.tensor(0).float().to(self.args.device)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_reg += self.lambda1 * torch.norm(param.to(self.args.device), 1)

        # Compute total loss
        loss = torch.mean(recon_loss) + torch.mean(kld) + l1_reg

        return (
            loss,
            torch.mean(nll).item(),
            torch.mean(mae).item(),
            torch.mean(kld).item(),
            l1_reg.item()
        )
    
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Applies higher weights to hard examples and lower weights to easy examples.
    """
    
    def __init__(
        self,
        lambda1: float,
        device: torch.device,
        alpha: float = 1.0,
        gamma: float = 0.0,
        logits: bool = False,
        reduce: bool = True
    ):
        """
        Args:
            lambda1: L1 regularization weight
            device: Computing device
            alpha: Weighting factor
            gamma: Focusing parameter
            logits: Whether inputs are logits
            reduce: Whether to reduce loss
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device
        self.lambda1 = torch.tensor(lambda1).to(device)

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            model: Neural network model
            inputs: Predicted values
            targets: Target values

        Returns:
            Focal loss value
        """
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy(
                inputs, targets, reduction='none'
            )

        # Compute probabilities
        pt = torch.exp(-bce_loss)
        
        # Compute focal weights
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply weights to BCE loss
        focal_loss = focal_weight * bce_loss

        # Add L1 regularization
        l1_reg = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_reg += torch.norm(param.to(self.device), 1)

        # Compute final loss
        loss = torch.mean(focal_loss)

        return loss


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Useful for segmentation tasks with imbalanced classes.
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        """
        Args:
            weight: Optional tensor of weights for BCE loss
        """
        super().__init__()
        self.bcelogits = nn.BCEWithLogitsLoss(weight=weight)

    def forward(
        self,
        y_score: torch.Tensor,
        y_out: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined Dice and BCE loss.

        Args:
            y_score: Predicted scores
            y_out: Predicted binary outputs
            targets: Target values
            smooth: Smoothing factor for Dice loss

        Returns:
            Tuple of (BCE loss, Combined loss)
        """
        # Compute BCE loss
        bce = self.bcelogits(y_out, targets)

        # Prepare inputs for Dice loss
        y_score = y_score.view(-1)
        targets = targets.view(-1)

        # Compute Dice loss components
        intersection = (y_score * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            y_score.sum() + targets.sum() + smooth
        )

        # Combine losses
        combined_loss = bce + dice_loss

        return bce, combined_loss

class VRNNLoss(nn.Module):
    """
    Loss function for Variational RNN.
    Combines reconstruction loss with KL divergence.
    """
    
    def __init__(
        self,
        lambda1: float,
        device: torch.device,
        isreconmsk: bool = True
    ):
        """
        Args:
            lambda1: L1 regularization weight
            device: Computing device
            isreconmsk: Whether to use reconstruction mask
        """
        super().__init__()
        self.lambda1 = torch.tensor(lambda1).to(device)
        self.device = device
        self.isreconmsk = isreconmsk

    def _kld_gauss(
        self,
        mean_1: torch.Tensor,
        std_1: torch.Tensor,
        mean_2: torch.Tensor,
        std_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian distributions.

        Args:
            mean_1, std_1: Parameters of first distribution
            mean_2, std_2: Parameters of second distribution

        Returns:
            KL divergence value
        """
        kld = (std_2 - std_1 + (torch.exp(std_1) + (mean_1 - mean_2).pow(2)) /
               torch.exp(std_2) - 1)
        return 0.5 * torch.sum(kld, 1)

    def forward(
        self,
        model: nn.Module,
        all_prior_mean: torch.Tensor,
        all_prior_std: torch.Tensor,
        all_x: torch.Tensor,
        all_enc_mean: torch.Tensor,
        all_enc_std: torch.Tensor,
        all_dec_mean: torch.Tensor,
        all_dec_std: torch.Tensor,
        msk: torch.Tensor,
        eval_x: torch.Tensor,
        eval_msk: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute VRNN loss.

        Args:
            model: Neural network model
            all_prior_mean: Prior distribution means
            all_prior_std: Prior distribution standard deviations
            all_x: Input tensors
            all_enc_mean: Encoder means
            all_enc_std: Encoder standard deviations
            all_dec_mean: Decoder means
            all_dec_std: Decoder standard deviations
            msk: Missing value mask
            eval_x: Evaluation tensor
            eval_msk: Evaluation mask
            beta: KL divergence weight

        Returns:
            Total loss value
        """
        kld_loss = 0
        nll_loss = 0
        mae_loss = 0

        for t in range(len(all_x)):
            # KL Divergence
            kld_loss += beta * self._kld_gauss(
                all_enc_mean[t],
                all_enc_std[t],
                all_prior_mean[t],
                all_prior_std[t]
            )

            if self.isreconmsk:
                # Masked reconstruction loss
                mu = all_dec_mean[t] * msk[:, t, :]
                std = (all_dec_std[t] * msk[:, t, :]).mul(0.5).exp_()

                # Create covariance matrices
                cov = []
                for vec in std:
                    cov.append(torch.diag(vec))
                cov = torch.stack(cov)

                # Compute negative log likelihood
                nll_loss += -MultivariateNormal(mu, cov).log_prob(
                    all_x[t] * msk[:, t, :]
                ).sum()

                # Compute MAE loss
                mae_loss += torch.abs(
                    all_dec_mean[t][eval_msk[:, t, :] == 1] -
                    eval_x[:, t, :][eval_msk[:, t, :] == 1]
                ).sum()
            else:
                # Unmasked losses
                nll_loss += -Normal(
                    all_dec_mean[t],
                    all_dec_std[t].mul(0.5).exp_()
                ).log_prob(all_x[t]).sum(1)
                
                mae_loss += torch.abs(
                    all_dec_mean[t] - all_x[t]
                ).sum(1)

        # Combine losses
        if self.isreconmsk:
            loss = kld_loss.mean() + nll_loss / len(kld_loss)
        else:
            loss = torch.mean(kld_loss + mae_loss + nll_loss)

        return loss

class DiceBCE_VariationalELBO(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCE_VariationalELBO, self).__init__()
        # self.dicebceLoss = DiceBCELoss()

    def forward(self, mll, output, y):

        likelihood_samples = mll.likelihood._draw_likelihood_samples(output)
        y_out = likelihood_samples.probs.mean(0).argmax(-1)
        y_score = likelihood_samples.probs.mean(0).max(-1).values

        # Dice_BCE_loss = self.dicebceLoss(y_score, y_out, y)

        res = likelihood_samples.log_prob(y).mean(dim=0).sum(-1)

        num_batch = output.event_shape[0]
        log_likelihood = res.div(num_batch)
        kl_divergence = mll.model.variational_strategy.kl_divergence().div(mll.num_data / mll.beta)
        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in mll.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in mll.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(mll.num_data))

        if mll.combine_terms:
            return -(log_likelihood - kl_divergence + log_prior - added_loss), y_out, y_score
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss, y_out, y_score
            else:
                return log_likelihood, kl_divergence, log_prior, y_out, y_score

class VRNNLoss(nn.Module):
    """
    Loss function for Variational RNN.
    Combines reconstruction loss with KL divergence.
    """
    
    def __init__(
        self,
        lambda1: float,
        device: torch.device,
        isreconmsk: bool = True
    ):
        """
        Args:
            lambda1: L1 regularization weight
            device: Computing device
            isreconmsk: Whether to use reconstruction mask
        """
        super().__init__()
        self.lambda1 = torch.tensor(lambda1).to(device)
        self.device = device
        self.isreconmsk = isreconmsk

    def _kld_gauss(
        self,
        mean_1: torch.Tensor,
        std_1: torch.Tensor,
        mean_2: torch.Tensor,
        std_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian distributions.

        Args:
            mean_1, std_1: Parameters of first distribution
            mean_2, std_2: Parameters of second distribution

        Returns:
            KL divergence value
        """
        kld = (std_2 - std_1 + (torch.exp(std_1) + (mean_1 - mean_2).pow(2)) /
               torch.exp(std_2) - 1)
        return 0.5 * torch.sum(kld, 1)

    def forward(
        self,
        model: nn.Module,
        all_prior_mean: torch.Tensor,
        all_prior_std: torch.Tensor,
        all_x: torch.Tensor,
        all_enc_mean: torch.Tensor,
        all_enc_std: torch.Tensor,
        all_dec_mean: torch.Tensor,
        all_dec_std: torch.Tensor,
        msk: torch.Tensor,
        eval_x: torch.Tensor,
        eval_msk: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute VRNN loss.

        Args:
            model: Neural network model
            all_prior_mean: Prior distribution means
            all_prior_std: Prior distribution standard deviations
            all_x: Input tensors
            all_enc_mean: Encoder means
            all_enc_std: Encoder standard deviations
            all_dec_mean: Decoder means
            all_dec_std: Decoder standard deviations
            msk: Missing value mask
            eval_x: Evaluation tensor
            eval_msk: Evaluation mask
            beta: KL divergence weight

        Returns:
            Total loss value
        """
        kld_loss = 0
        nll_loss = 0
        mae_loss = 0

        for t in range(len(all_x)):
            # KL Divergence
            kld_loss += beta * self._kld_gauss(
                all_enc_mean[t],
                all_enc_std[t],
                all_prior_mean[t],
                all_prior_std[t]
            )

            if self.isreconmsk:
                # Masked reconstruction loss
                mu = all_dec_mean[t] * msk[:, t, :]
                std = (all_dec_std[t] * msk[:, t, :]).mul(0.5).exp_()

                # Create covariance matrices
                cov = []
                for vec in std:
                    cov.append(torch.diag(vec))
                cov = torch.stack(cov)

                # Compute negative log likelihood
                nll_loss += -MultivariateNormal(mu, cov).log_prob(
                    all_x[t] * msk[:, t, :]
                ).sum()

                # Compute MAE loss
                mae_loss += torch.abs(
                    all_dec_mean[t][eval_msk[:, t, :] == 1] -
                    eval_x[:, t, :][eval_msk[:, t, :] == 1]
                ).sum()
            else:
                # Unmasked losses
                nll_loss += -Normal(
                    all_dec_mean[t],
                    all_dec_std[t].mul(0.5).exp_()
                ).log_prob(all_x[t]).sum(1)
                
                mae_loss += torch.abs(
                    all_dec_mean[t] - all_x[t]
                ).sum(1)

        # Combine losses
        if self.isreconmsk:
            loss = kld_loss.mean() + nll_loss / len(kld_loss)
        else:
            loss = torch.mean(kld_loss + mae_loss + nll_loss)

        return loss


class AsymSimilarityLoss(nn.Module):
    """
    Asymmetric Similarity Loss.
    Applies different weights to positive and negative examples.
    """
    
    def __init__(
        self,
        beta: float,
        lambda1: float,
        device: torch.device
    ):
        """
        Args:
            beta: Asymmetry parameter
            lambda1: L1 regularization weight
            device: Computing device
        """
        super().__init__()
        self.beta = beta
        self.lambda1 = lambda1
        self.device = device

    def forward(
        self,
        model: nn.Module,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric similarity loss.

        Args:
            model: Neural network model
            y_pred: Predicted values
            y: Target values

        Returns:
            Loss value
        """
        # Compute numerator
        nom = (1 + self.beta**2) * torch.sum(y_pred * y.float())
        
        # Compute denominator components
        denom = (
            (1 + self.beta**2) * torch.sum(y_pred * y.float()) +
            (self.beta**2 * torch.sum((1-y_pred) * y.float())) +
            torch.sum(y_pred * (1 - y).float())
        )
        
        # Compute similarity loss
        asym_sim_loss = nom / denom

        return asym_sim_loss


def test_losses():
    """
    Test loss functions with dummy data.
    """
    # Create dummy data
    output = torch.randint(0, 10, size=(10,)).float()
    score = torch.sigmoid(output)
    target = torch.randint(0, 10, size=(10,)).float()
    
    # Test DiceBCELoss
    dbce = DiceBCELoss()
    bce_loss, combined_loss = dbce(score, output, target)
    
    print(f"BCE Loss: {bce_loss.item():.4f}")
    print(f"Combined Loss: {combined_loss.item():.4f}")

if __name__ == '__main__':
    test_losses()

