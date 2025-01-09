# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import math

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true.long(), num_classes=2).float()
        return self.loss_fn(y_pred, y_true)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:, :X.size(1)]
        return self.dropout(X)

class TransformerClassifier(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes=2, 
                 hidden_dim=64, 
                 nhead=8, 
                 num_encoder_layers=2, 
                 dropout=0.1,
                 window_size=10):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        self.input_linear = nn.Linear(num_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=window_size)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        if len(X.shape) == 2:
            X = X.unsqueeze(1)
        
        X = X.float()
        X = self.input_linear(X)
        X = self.pos_encoder(X)
        X = self.transformer_encoder(X)
        X = X.mean(dim=1)
        X = self.dropout(X)
        logits = self.fc(X)
        
        return logits

class MLPClassifierModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(MLPClassifierModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        X = self.fc1(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X  # 返回 logits [batch_size, 2]
