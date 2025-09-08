"""
Advanced PyTorch Deep Learning Models for NBA Props Prediction
Implements GANs, Variational Autoencoders, and Graph Neural Networks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

class NBAPropDataset(Dataset):
    """Custom dataset for NBA props data"""
    def __init__(self, sequences, targets, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, target

class AdvancedLSTMModel(nn.Module):
    """Advanced LSTM with attention mechanism"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(AdvancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        lstm_out = lstm_out.permute(1, 0, 2)  # seq_len, batch, features
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # batch, seq_len, features
        
        # Layer normalization
        attn_out = self.layer_norm(attn_out + lstm_out.permute(1, 0, 2))
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class TransformerModel(nn.Module):
    """Pure Transformer model for sequence prediction"""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects seq_len first
        x = x.permute(1, 0, 2)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=0)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for player relationship modeling"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=0.2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.2))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Final layer
        self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Final conv layer
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        output = self.classifier(x)
        
        return output

class VariationalAutoencoder(nn.Module):
    """VAE for learning player performance representations"""
    def __init__(self, input_dim, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, predict=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        
        if predict:
            prediction = self.predictor(z)
            return reconstructed, mu, logvar, prediction
        
        return reconstructed, mu, logvar

class GenerativeAdversarialNetwork:
    """GAN for generating synthetic training data"""
    def __init__(self, input_dim, latent_dim=100):
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)

class Generator(nn.Module):
    """Generator network for GAN"""
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.network(z)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class PyTorchPredictor:
    """Main PyTorch predictor class"""
    def __init__(self, model_dir='ai_models/pytorch'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Models
        self.lstm_model = None
        self.transformer_model = None
        self.gnn_model = None
        self.vae_model = None
        self.gan = None
        
        # Scaler
        self.scaler = StandardScaler()
    
    def create_models(self, input_size):
        """Create all PyTorch models"""
        self.lstm_model = AdvancedLSTMModel(input_size).to(self.device)
        self.transformer_model = TransformerModel(input_size).to(self.device)
        self.gnn_model = GraphNeuralNetwork(input_size).to(self.device)
        self.vae_model = VariationalAutoencoder(input_size).to(self.device)
        self.gan = GenerativeAdversarialNetwork(input_size)
        
        print("✅ All PyTorch models created successfully!")
    
    def train_models(self, train_data, val_data, epochs=100):
        """Train all models with advanced techniques"""
        print("🚀 Starting PyTorch training pipeline...")
        
        # Prepare data loaders
        train_dataset = NBAPropDataset(train_data['X'], train_data['y'])
        val_dataset = NBAPropDataset(val_data['X'], val_data['y'])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training configurations
        models = {
            'lstm': self.lstm_model,
            'transformer': self.transformer_model,
            'vae': self.vae_model
        }
        
        optimizers = {
            'lstm': optim.AdamW(self.lstm_model.parameters(), lr=0.001, weight_decay=0.01),
            'transformer': optim.AdamW(self.transformer_model.parameters(), lr=0.001, weight_decay=0.01),
            'vae': optim.AdamW(self.vae_model.parameters(), lr=0.001, weight_decay=0.01)
        }
        
        schedulers = {
            name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for name, opt in optimizers.items()
        }
        
        # Training loop
        training_history = {name: {'train_loss': [], 'val_loss': [], 'val_acc': []} 
                          for name in models.keys()}
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            for model_name, model in models.items():
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizers[model_name].zero_grad()
                    
                    if model_name == 'vae':
                        recon, mu, logvar, pred = model(data, predict=True)
                        loss = self.vae_loss(recon, data, mu, logvar, pred, target)
                    else:
                        output = model(data)
                        loss = F.binary_cross_entropy(output.squeeze(), target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizers[model_name].step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        if model_name == 'vae':
                            recon, mu, logvar, pred = model(data, predict=True)
                            loss = self.vae_loss(recon, data, mu, logvar, pred, target)
                            predicted = (pred.squeeze() > 0.5).float()
                        else:
                            output = model(data)
                            loss = F.binary_cross_entropy(output.squeeze(), target)
                            predicted = (output.squeeze() > 0.5).float()
                        
                        val_loss += loss.item()
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                
                # Update learning rate
                schedulers[model_name].step()
                
                # Record metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total
                
                training_history[model_name]['train_loss'].append(avg_train_loss)
                training_history[model_name]['val_loss'].append(avg_val_loss)
                training_history[model_name]['val_acc'].append(val_accuracy)
                
                if epoch % 10 == 0:
                    print(f"{model_name}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save models
        self.save_models()
        
        return training_history
    
    def vae_loss(self, recon_x, x, mu, logvar, pred, target, beta=1.0):
        """VAE loss function with prediction loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Prediction loss
        pred_loss = F.binary_cross_entropy(pred.squeeze(), target)
        
        return recon_loss + beta * kl_loss + pred_loss
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = {}
        
        # LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(X_tensor).cpu().numpy()
            predictions['lstm'] = lstm_pred
        
        # Transformer prediction
        self.transformer_model.eval()
        with torch.no_grad():
            transformer_pred = self.transformer_model(X_tensor).cpu().numpy()
            predictions['transformer'] = transformer_pred
        
        # VAE prediction
        self.vae_model.eval()
        with torch.no_grad():
            _, _, _, vae_pred = self.vae_model(X_tensor, predict=True)
            predictions['vae'] = vae_pred.cpu().numpy()
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            0.4 * predictions['lstm'] +
            0.4 * predictions['transformer'] +
            0.2 * predictions['vae']
        )
        
        predictions['ensemble'] = ensemble_pred
        
        # Calculate confidence based on prediction variance
        pred_array = np.array([predictions['lstm'], predictions['transformer'], predictions['vae']])
        confidence = 1 - np.var(pred_array, axis=0)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'final_prediction': ensemble_pred
        }
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data using GAN"""
        self.gan.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.gan.latent_dim).to(self.device)
            synthetic_data = self.gan.generator(z).cpu().numpy()
        
        return synthetic_data
    
    def save_models(self):
        """Save all trained models"""
        torch.save(self.lstm_model.state_dict(), 
                  os.path.join(self.model_dir, 'lstm_model.pth'))
        torch.save(self.transformer_model.state_dict(), 
                  os.path.join(self.model_dir, 'transformer_model.pth'))
        torch.save(self.gnn_model.state_dict(), 
                  os.path.join(self.model_dir, 'gnn_model.pth'))
        torch.save(self.vae_model.state_dict(), 
                  os.path.join(self.model_dir, 'vae_model.pth'))
        
        # Save scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_models(self, input_size):
        """Load pre-trained models"""
        try:
            self.create_models(input_size)
            
            self.lstm_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'lstm_model.pth'))
            )
            self.transformer_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'transformer_model.pth'))
            )
            self.gnn_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'gnn_model.pth'))
            )
            self.vae_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'vae_model.pth'))
            )
            
            # Load scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ All PyTorch models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False