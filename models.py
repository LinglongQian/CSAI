"""
CSAI Model Implementation

This module contains implementations of CSAI and related baseline models for time series imputation.
Key components include:
- Basic building blocks (Decay, FeatureRegression)
- Transformer components for attention
- RNN-based models (BRITS, GRU-D, etc.)
- The main CSAI architecture
"""

import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import copy
import pandas as pd
from losses import SVAELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# Type hints
Tensor = torch.Tensor
TensorTuple = Tuple[Tensor, ...]

class BasicModules(nn.Module):
    """Base class for all model components."""
    
    def reset_parameters(self):
        """Initialize model parameters using uniform distribution."""
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stdv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stdv, stdv)

class FeatureRegression(BasicModules):
    """Feature-wise regression module for cross-sectional correlations."""
    
    def __init__(self, input_size: int):
        """
        Args:
            input_size: Dimension of input features
        """
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        
        # Create mask for diagonal elements
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feature regression.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Tensor of shape [batch_size, input_size]
        """
        # Mask diagonal elements and apply linear transformation
        return F.linear(x, self.W * self.m, self.b)

class Decay(BasicModules):
    """Temporal decay module for handling time gaps."""
    
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        """
        Args:
            input_size: Dimension of input features
            output_size: Dimension of output features
            diag: If True, applies decay only to diagonal elements
        """
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size, "Input and output sizes must match for diagonal decay"
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        
        self.reset_parameters()

    def forward(self, d: Tensor) -> Tensor:
        """
        Compute decay factors.
        
        Args:
            d: Time gaps tensor
            
        Returns:
            Decay factors tensor
        """
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        return torch.exp(-gamma)
    
class Decay_obs(BasicModules):
    """
    Observable temporal decay module that adapts to the sign of time differences.
    This module learns a decay function that considers both positive and negative time 
    differences to handle different types of temporal dependencies.
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the observable decay module.
        
        Args:
            input_size: Dimension of input features
            output_size: Dimension of output features
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.reset_parameters()

    def forward(self, delta_diff: Tensor) -> Tensor:
        """
        Compute decay weights based on time differences.
        
        When delta_diff is negative (observation is recent), weight tends to 1.
        When delta_diff is positive (observation is old), weight tends to 0.
        
        Args:
            delta_diff: Tensor of time differences between current time and last observation
                Shape: [batch_size, feature_dim]
                
        Returns:
            Tensor of decay weights with same shape as input
                Values are between 0 and 1, where:
                - Values closer to 1 indicate more relevance (recent observations)
                - Values closer to 0 indicate less relevance (old observations)
        """
        # Get sign information to determine direction of decay
        sign = torch.sign(delta_diff)
        
        # Calculate raw weights through linear layer
        weight_diff = self.linear(delta_diff)
        
        # Split into positive and negative components for more stable training
        positive_part = F.relu(weight_diff)
        negative_part = F.relu(-weight_diff)
        
        # Combine components based on sign
        weight_diff = positive_part + negative_part
        weight_diff = sign * weight_diff
        
        # Squeeze to [-1, 1] range
        weight_diff = torch.tanh(weight_diff)
        
        # Transform to [0, 1] range where:
        # - weight → 1 for recent observations (negative delta_diff)
        # - weight → 0 for old observations (positive delta_diff)
        weight = 0.5 * (1 - weight_diff)
        
        return weight

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer components."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class rits(BasicModules):
    def __init__(self, args, dropout=0.25):
        super().__init__()
        self.args = args
        
        self.input_size = self._get_input_size()
        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.lstm = nn.LSTMCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def _get_input_size(self) -> int:
        """Determine input size based on dataset."""
        dataset_sizes = {
            'physionet': 35,
            'mimic_59f': 59,
            'eicu': 20
        }
        return dataset_sizes.get(self.args.dataset)

    def forward(self, x, mask, deltas, h=None, get_y=False):
        # Get dimensionality
        [B, T, V] = x.shape
        
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)
        if c == None:
            c = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)
            
        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        for t in range(T):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            
            # history based estimation
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h, c = self.lstm(input_t, (h, c))

            # Keep the imputation
            Hiddens.append(h.unsqueeze(dim=1))
        Hiddens = torch.cat(Hiddens, dim=1)

        if (self.args.task == 'C') and (get_y == True):
            y_out = self.classification(self.dropout(h))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'xloss':x_loss, 'hidden_state':Hiddens, 'y_out':y_out, 'y_score':y_score}

        return ret

class brits(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(brits, self).__init__()
        self.args = args
        self.model_f = rits(args=self.args)
        self.model_b = rits(args=self.args)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)

        ret_f = self.model_f(x, m, d_f, get_y=self.get_y)
        # Set data to be backward
        x_b = x.flip(dims=[1])
        m_b = m.flip(dims=[1])
        ret_b = self.model_b(x_b, m_b, d_b, get_y=self.get_y)

        # Averaging the imputations and prediction
        x_imp = (ret_f['imputation'] + ret_b['imputation'].flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(ret_f['imputation'] - ret_b['imputation'].flip(dims=[1])).mean() * 1e-1

        # average the regression loss
        xreg_loss = ret_f['xloss'] + ret_b['xloss']

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'y_out_f':ret_f['y_out'], 'y_score_f':ret_f['y_score'], 'y_out_b':ret_b['y_out'], 'y_score_b':ret_b['y_score']}
        return ret

class rits_gru(BasicModules):
    def __init__(self, args, dropout=0.25):
        super().__init__()
        self.args = args
        
        self.input_size = self._get_input_size()

        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def _get_input_size(self) -> int:
        """Determine input size based on dataset."""
        dataset_sizes = {
            'physionet': 35,
            'mimic_59f': 59,
            'eicu': 20
        }
        return dataset_sizes.get(self.args.dataset)
    
    def forward(self, x, mask, deltas, h=None, get_y=False):
        # Get dimensionality
        [B, T, V] = x.shape
        
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        for t in range(T):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            
            # history based estimation
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h = self.gru(input_t, h)

            # Keep the imputation
            Hiddens.append(h.unsqueeze(dim=1))
        Hiddens = torch.cat(Hiddens, dim=1)

        if (self.args.task == 'C') and (get_y == True):
            y_out = self.classification(self.dropout(h))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'xloss':x_loss, 'hidden_state':Hiddens, 'y_out':y_out, 'y_score':y_score}

        return ret

class brits_gru(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(brits_gru, self).__init__()
        self.args = args
        self.model_f = rits_gru(args=self.args)
        self.model_b = rits_gru(args=self.args)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)

        ret_f = self.model_f(x, m, d_f, get_y=self.get_y)
        # Set data to be backward
        x_b = x.flip(dims=[1])
        m_b = m.flip(dims=[1])
        ret_b = self.model_b(x_b, m_b, d_b, get_y=self.get_y)

        # Averaging the imputations and prediction
        x_imp = (ret_f['imputation'] + ret_b['imputation'].flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(ret_f['imputation'] - ret_b['imputation'].flip(dims=[1])).mean() * 1e-1

        # average the regression loss
        xreg_loss = ret_f['xloss'] + ret_b['xloss']

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'y_out_f':ret_f['y_out'], 'y_score_f':ret_f['y_score'], 'y_out_b':ret_b['y_out'], 'y_score_b':ret_b['y_score']}
        return ret

class CSAI(BasicModules):
    """Conditional Self-Attention Imputation model."""
    
    def __init__(self, args, dropout: float = 0.25, medians_df: Optional[Dict] = None):
        """
        Initialize CSAI model.
        
        Args:
            args: Configuration object
            dropout: Dropout probability
            medians_df: Dictionary of median values for each feature
        """
        super().__init__()
        self.args = args
        
        # Set input size based on dataset
        self.input_size = self._get_input_size()
        self.hidden_size = args.hiddens
        self.dropout = dropout

        # Initialize median values
        if medians_df is not None:
            self.medians_tensor = torch.tensor(list(medians_df.values())).float()
        else:
            self.medians_tensor = torch.zeros(self.input_size).float()
            
        # Initialize model components
        self._init_components()

    def _get_input_size(self) -> int:
        """Determine input size based on dataset."""
        dataset_sizes = {
            'physionet': 35,
            'mimic_59f': 59,
            'eicu': 20
        }
        return dataset_sizes.get(self.args.dataset)

    def _init_components(self):
        """Initialize all model components."""
        # Decay and regression components
        self.temp_decay_h = Decay(self.input_size, self.hidden_size, diag=False)
        self.temp_decay_x = Decay(self.input_size, self.input_size, diag=True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.weighted_obs = Decay_obs(self.input_size, self.input_size)

        # Attention components
        self.pos_encoder = PositionalEncoding(self.args.step_channels)
        self.input_projection = nn.Conv1d(self.input_size, self.args.step_channels, 1)
        self.output_projection1 = nn.Conv1d(self.args.step_channels, self.hidden_size, 1)
        self.output_projection2 = nn.Conv1d(self.args.hours*2, 1, 1)
        self.time_layer = self._get_transformer_encoder()
        
        # Other components
        self.dropout = nn.Dropout(self.dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        
        self.reset_parameters()

    def _get_transformer_encoder(self) -> nn.TransformerEncoder:
        """Create transformer encoder layer."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.step_channels,
            nhead=8,
            dim_feedforward=64,
            activation="gelu"
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: Tensor, mask: Tensor, deltas: Tensor, last_obs: Tensor, 
                h: Optional[Tensor] = None, get_y: bool = True) -> Dict[str, Tensor]:
        """
        Forward pass of CSAI model.
        
        Args:
            x: Input tensor
            mask: Missing value mask
            deltas: Time gap tensor
            last_obs: Last observed values
            h: Hidden state (optional)
            get_y: Whether to compute classification output
            
        Returns:
            Dictionary containing:
                - imputation: Imputed values
                - xloss: Imputation loss
                - hidden_state: Hidden states
                - y_out: Classification logits (if get_y=True)
                - y_score: Classification probabilities (if get_y=True)
        """
        # Get batch size
        batch_size = x.size(0)

        medians = self.medians_tensor.unsqueeze(0).repeat(batch_size, 1).to(x.device)

        decay_factor = self.weighted_obs(deltas - medians.unsqueeze(1))

        # Initialize hidden state if not provided
        if h is None:
            h = self._initialize_hidden(last_obs, decay_factor)
            
        # Prepare storage for outputs
        x_loss = 0
        x_imp = x.clone()
        hidden_states = []

        # Process each timestep
        for t in range(x.size(1)):
            # Get current timestep data
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]
            
            # Apply temporal decay
            h = self._apply_temporal_decay(h, d_t)
            
            # Generate imputations
            x_imp[:, t, :], x_loss_t = self._impute_timestep(x_t, m_t, d_t, h)
            x_loss += x_loss_t
            
            # Update hidden state
            h = self._update_hidden_state(x_imp[:, t, :], m_t, h)
            hidden_states.append(h.unsqueeze(1))

        # Combine all hidden states
        hidden_states = torch.cat(hidden_states, dim=1)
        
        # Prepare output dictionary
        output = {
            'imputation': x_imp,
            'xloss': x_loss,
            'hidden_state': hidden_states,
            'y_out': 0,
            'y_score': 0
        }
        
        # Add classification outputs if requested
        if self.args.task == 'C' and get_y:
            y_out = self.classification(self.dropout(h))
            y_score = torch.sigmoid(y_out)
            output.update({'y_out': y_out, 'y_score': y_score})
            
        return output

    def _initialize_hidden(self, last_obs: Tensor, decay_factor: Tensor) -> Tensor:
        """Initialize hidden state using attention mechanism."""
        # Project inputs
        last_obs_proj = self._project_and_encode(last_obs)
        decay_factor_proj = self._project_and_encode(decay_factor)
        
        # Combine and apply attention
        combined = torch.cat([last_obs_proj, decay_factor_proj], dim=1)
        attended = self.time_layer(combined)
        
        # Project to hidden dimension
        h1 = self.output_projection1(attended.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = self.output_projection2(h1).squeeze()
        
        return hidden

    def _project_and_encode(self, x: Tensor) -> Tensor:
        """Project and apply positional encoding to input."""
        x = self.input_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

    def _apply_temporal_decay(self, h: Tensor, d_t: Tensor) -> Tensor:
        """Apply temporal decay to hidden state."""
        gamma_h = self.temp_decay_h(d_t)
        return h * gamma_h

    def _impute_timestep(self, x_t: Tensor, m_t: Tensor, d_t: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate imputation for single timestep."""
        # Generate history-based estimation
        x_h = self.hist(h)
        x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)
        
        # Generate feature-based estimation
        xu = self.feat_reg_v(x_r_t)
        gamma_x = self.temp_decay_x(d_t)
        
        # Combine estimates
        beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
        x_comb_t = beta * xu + (1 - beta) * x_h
        
        # Calculate loss
        x_loss = torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)
        
        # Final imputation
        x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)
        
        return x_imp_t, x_loss

    def _update_hidden_state(self, x_t: Tensor, m_t: Tensor, h: Tensor) -> Tensor:
        """Update hidden state using GRU cell."""
        input_t = torch.cat([x_t, m_t], dim=1)
        return self.gru(input_t, h)

class bcsai(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(bcsai, self).__init__()
        self.args = args
        self.model_f = CSAI(args=self.args, medians_df=medians_df)
        self.model_b = CSAI(args=self.args, medians_df=medians_df)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)
        last_obs_f = xdata['last_obs_f'].to(self.args.device)
        last_obs_b = xdata['last_obs_b'].to(self.args.device)

        ret_f = self.model_f(x, m, d_f, last_obs_f, get_y=self.get_y)
        # Set data to be backward
        x_b = x.flip(dims=[1])
        m_b = m.flip(dims=[1])
        ret_b = self.model_b(x_b, m_b, d_b, last_obs_b, get_y=self.get_y)

        # Averaging the imputations and prediction
        x_imp = (ret_f['imputation'] + ret_b['imputation'].flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(ret_f['imputation'] - ret_b['imputation'].flip(dims=[1])).mean() * 1e-1

        # average the regression loss
        xreg_loss = ret_f['xloss'] + ret_b['xloss']

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'y_out_f':ret_f['y_out'], 'y_score_f':ret_f['y_score'], 'y_out_b':ret_b['y_out'], 'y_score_b':ret_b['y_score']}
        return ret

class gru_d(BasicModules):
    def __init__(self, args, dropout=0.25, medians_df=None, get_y=False):
        super().__init__()
        self.args = args
        self.input_size = self._get_input_size()

        self.get_y = get_y
        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def _get_input_size(self) -> int:
        """Determine input size based on dataset."""
        dataset_sizes = {
            'physionet': 35,
            'mimic_59f': 59,
            'eicu': 20
        }
        return dataset_sizes.get(self.args.dataset)
    
    def forward(self, xdata, meanset, direct='forward', hidden=None):
        x = xdata['values'].to(self.args.device)
        mask = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            mask = mask.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)

        meanset = torch.tensor(meanset).to(self.args.device)
        
        x_original = copy.deepcopy(x)
        x_original[mask==0] = np.nan
        x_forward = [pd.DataFrame(x_original[i,:,:].cpu().numpy()).fillna(method='ffill').fillna(0.0).values for i in range(x_original.size(0))]
        x_forward = torch.from_numpy(np.array(x_forward)).to(self.args.device)

        [B, T, V] = x.shape

        if hidden == None:
            hidden = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)
        
        x_loss = 0
        x_imp = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            d_t = deltas[:, t, :]
            f_t = x_forward[:, t, :]

            gamma_h = self.temp_decay_h(d_t)
            hidden = hidden * gamma_h

            gamma_x = self.temp_decay_x(d_t)
            x_u = gamma_x * f_t + (1 - gamma_x) *  meanset

            x_loss += torch.sum(torch.abs(x_t - x_u) * m_t) / (torch.sum(m_t) + 1e-5)
            
            x_h = m_t * x_t + (1 - m_t) * x_u
            inputs = torch.cat([x_h, m_t], dim = 1).float()

            hidden = self.gru(inputs, hidden)

            x_imp.append(x_h.unsqueeze(dim = 1))

        x_imp = torch.cat(x_imp, dim = 1)

        if (self.args.task == 'C') and (self.get_y == True):
            y_out = self.classification(self.dropout(hidden))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'loss_consistency':0, 'loss_regression':x_loss, 'y_out_f':y_out, 'y_score_f':y_score, 'y_out_b':y_out, 'y_score_b':y_score}
        return ret

class m_rnn(BasicModules):
    def __init__(self, args, dropout=0.25, medians_df=None, get_y=False):
        super().__init__()
        self.args = args
        
        self.input_size = self._get_input_size()
        self.hidden_size = self.args.hiddens
        self.get_y = get_y
        self.hist_reg = nn.Linear(self.hidden_size * 2, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.imputation = nn.Linear(self.input_size, self.input_size)

        self.rnn_cell = nn.GRUCell(self.input_size * 3, self.hidden_size)
        self.pred_rnn = nn.GRU(self.input_size, self.hidden_size, batch_first = True)

        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)

        self.reset_parameters()

    def _get_input_size(self) -> int:
        """Determine input size based on dataset."""
        dataset_sizes = {
            'physionet': 35,
            'mimic_59f': 59,
            'eicu': 20
        }
        return dataset_sizes.get(self.args.dataset)

    def get_hidden(self, xdata, direct, hidden=None):
        x = xdata['values'].to(self.args.device)
        masks = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            masks = masks.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)
        [B, T, V] = x.shape
        hiddens = []
        if hidden == None:
            hidden = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        for t in range(T):
            hiddens.append(hidden)
            x_t = x[:, t, :]
            m_t = masks[:, t, :]
            d_t = deltas[:, t, :]
            inputs = torch.cat([x_t, m_t, d_t], dim = 1)
            hidden = self.rnn_cell(inputs, hidden)
        return hiddens

    def forward(self, xdata, direct='forward'):

        hidden_forward = self.get_hidden(xdata, 'forward')
        hidden_backward = self.get_hidden(xdata, 'backward')[::-1]

        x = xdata['values'].to(self.args.device)
        masks = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            masks = masks.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)

        [B, T, V] = x.shape
        x_loss = 0
        x_imp = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = masks[:, t, :]
            d_t = deltas[:, t, :]

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim = 1)

            x_v = self.hist_reg(h)
            x_u = self.feat_reg(x_t)
            x_h = x_u + self.weight_combine(torch.cat([x_v, m_t], dim = 1))
            x_imp_t = self.imputation(x_h)

            x_loss += torch.sum(torch.abs(x_t - x_imp_t) * m_t) / (torch.sum(m_t) + 1e-5)

            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_imp_t)
            x_imp.append(x_imp_t.unsqueeze(dim = 1))

        x_imp = torch.cat(x_imp, dim = 1)

        if (self.args.task == 'C') and (self.get_y == True):
            out, h = self.pred_rnn(x_imp)
            y_out = self.classification(self.dropout(h.squeeze()))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0
        
        ret = {'imputation':x_imp, 'loss_consistency':0, 'loss_regression':x_loss, 'y_out_f':y_out, 'y_score_f':y_score, 'y_out_b':y_out, 'y_score_b':y_score}
        return ret

class VAE(BasicModules):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.hiddens = self.args.vae_hiddens

        # Encoder
        self.enc = nn.Sequential()
        for i in range(len(self.hiddens)-2):
            self.enc.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i+1]))
            self.enc.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i+1]))
            self.enc.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.enc.add_module("tanh_%d" % i, nn.Tanh())
        self.enc_mu = nn.Linear(self.hiddens[-2], self.hiddens[-1])
        self.enc_logvar = nn.Linear(self.hiddens[-2], self.hiddens[-1])

        # Decoder
        self.dec = nn.Sequential()
        for i in range(len(self.hiddens))[::-1][:-2]:
            self.dec.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i-1]))
            self.dec.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i-1]))
            self.dec.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.dec.add_module("tanh_%d" % i, nn.Tanh())
        self.dec_mu = nn.Linear(self.hiddens[1], self.hiddens[0])
        self.dec_logvar = nn.Linear(self.hiddens[1], self.hiddens[0])

        self.reset_parameters()

    # Reparameterize
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):

        # Encoding
        e = self.enc(x)
        enc_mu = self.enc_mu(e)
        enc_logvar =self.enc_logvar(e)
        z = self.reparameterize(enc_mu, enc_logvar)

        # Decoding
        d = self.dec(z)
        dec_mu = self.dec_mu(d)
        dec_logvar = self.dec_logvar(d)
        x_hat = dec_mu

        return z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar

class RIN(BasicModules):
    def __init__(self, args):#
        super().__init__()
        self.args = args
        
        self.input_size = self._get_input_size()
        self.hidden_size = self.args.hiddens

        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.feat_reg_r = FeatureRegression(self.input_size)

        self.unc_flag = self.args.unc_flag
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Activate only for the model with uncertainty
        if self.args.unc_flag == 1:
            self.unc_decay = Decay(input_size=self.input_size, output_size=self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, x_hat, u, m, d, h=None, get_y=False):
        # Get dimensionality
        [B, T, _] = x.shape

        # Initialize Hidden weights
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        x_loss = 0
        # x_imp = torch.Tensor().cuda()
        x_imp = []
        xus = []
        xrs = []
        for t in range(T):
            x_t = x[:, t, :]
            x_hat_t = x_hat[:, t, :]
            u_t = u[:, t, :]
            d_t = d[:, t, :]
            m_t = m[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h

            # Regression
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            if self.args.unc_flag == 1:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar) * self.unc_decay(u_t)
            else:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar)

            xr = self.feat_reg_r(x_r_t)

            x_comb_t = self.conv1(torch.cat([xu.unsqueeze(1), xr.unsqueeze(1)], dim=1)).squeeze(1)
            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the the RNN
            input_t = torch.cat([x_imp_t, m_t], dim=1)

            # Feed into GRU cell, get the hiddens
            h = self.gru(input_t, h)

            # Keep the imputation
            x_imp.append(x_imp_t.unsqueeze(dim=1))
            xus.append(xu.unsqueeze(dim=1))
            xrs.append(xr.unsqueeze(dim=1))

        x_imp = torch.cat(x_imp, dim=1)
        xus = torch.cat(xus, dim=1)
        xrs = torch.cat(xrs, dim=1)

        # Get the output
        if (self.args.task == 'C') and (get_y == True):
            y_out = self.fc_out(h)
            y_score = self.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        return x_imp, y_out, y_score, x_loss, xus, xrs

class bvrin(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(bvrin, self).__init__()
        self.args = args
        self.vae = VAE(self.args)
        self.rin_f = RIN(self.args)
        self.rin_b = RIN(self.args)
        self.criterion_vae = SVAELoss(self.args)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)
        eval_x = xdata['evals'].to(self.args.device)
        eval_m = xdata['eval_masks'].to(self.args.device)
        y = xdata['labels'].to(self.args.device)
        
        [B, T, V] = x.shape
        # VAE
        rx = x.contiguous().view(-1, V)
        rm = m.contiguous().view(-1, V)
        z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar = self.vae(rx)
        unc = (m * torch.zeros(B, T, V).to(self.args.device)) + ((1 - m) * torch.exp(0.5 * dec_logvar).view(B, T, V))

        # RIN Forward
        x_imp_f, y_out_f, y_score_f, xreg_loss_f, _, _ = self.rin_f(x, x_hat.view(B, T, V), unc, m, d_f, get_y=self.get_y)

        # Set data to be backward
        x_b = x.flip(dims=[1])
        x_hat_b = x_hat.view(B, T, V).flip(dims=[1])
        unc_b = unc.flip(dims=[1])
        m_b = m.flip(dims=[1])

        # RIN Backward
        x_imp_b, y_out_b, y_score_b, xreg_loss_b, _, _ = self.rin_b(x_b, x_hat_b, unc_b, m_b, d_b, get_y=self.get_y)

        loss_vae, lossnll, lossmae, losskld, lossl1 = self.criterion_vae(self.vae, rx, eval_x.view(B*T, V), x_hat.view(B*T, V), rm, eval_m.view(B*T, V), enc_mu, enc_logvar, dec_mu, dec_logvar, phase='train')   

        # Averaging the imputations and prediction
        x_imp = (x_imp_f + x_imp_b.flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(x_imp_f - x_imp_b.flip(dims=[1])).mean() * 1e-1

        # Sum the regression loss
        xreg_loss = xreg_loss_f + xreg_loss_b

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'loss_vae':loss_vae, 'y_out_f':y_out_f, 'y_score_f':y_score_f, 'y_out_b':y_out_b, 'y_score_b':y_score_b}
        return ret