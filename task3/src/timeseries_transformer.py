import os
import logging
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # TensorBoard integration
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from shapely.wkt import loads
from tqdm import tqdm  # For progress bars
import argparse
from tqdm import tqdm
tqdm.pandas()  # Initialize tqdm for Pandas

# Set up concise logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Dataset and DataLoader for Time-Series
class MobilityDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Transformer Model for Time-Series Prediction
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_layers, output_dim):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, model_dim))  # Adjust sequence length
        self.transformer = nn.Transformer(d_model=model_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer(x, x)
        return self.fc(x[:, -1, :])  # Predict the next timestep

# Helper function to parse LINESTRING into coordinates
def parse_linestring(linestring):
    geom = loads(linestring)
    return list(geom.coords)

# Helper function to create sliding windows
def create_windowed_data(data, window_size):
    sequences, labels = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        labels.append(data[i+window_size])
    return np.array(sequences), np.array(labels)

# Preprocessing function
def preprocess_data(file_path, window_size):
    logging.info("Step 1: Loading and preprocessing data.")
    
    # Load data
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows of data.")
    
    # Parse geom to extract coordinates
    logging.info("Parsing LINESTRING data...")
    df['coords'] = df['geom'].progress_apply(parse_linestring)  # Uses tqdm.pandas for progress bar
    total_coords = sum(df['coords'].apply(len))
    logging.info(f"Extracted a total of {total_coords} coordinates from the dataset.")

    # Flatten trajectories into (x, y) sequences and normalize
    logging.info("Flattening and normalizing coordinates...")
    all_coords = np.vstack(df['coords'].values)
    scaler = MinMaxScaler()
    all_coords = scaler.fit_transform(all_coords)
    df['normalized_coords'] = df['coords'].progress_apply(lambda coords: scaler.transform(coords))
    
    # Create sliding windows for sequences
    logging.info("Creating sliding windows for sequences...")
    sequences = []
    for normalized_coords in tqdm(df['normalized_coords'], desc="Generating Sequences"):
        seq, _ = create_windowed_data(normalized_coords, window_size)
        sequences.extend(seq)
    sequences = np.array(sequences)
    
    # Generate labels (next coordinates)
    labels = sequences[:, -1, :]  # Last coordinate in each sequence is the label
    sequences = sequences[:, :-1, :]  # Remove last coordinate from sequences
    logging.info(f"Generated {len(sequences)} sequences with labels.")
    
    # Log sequence statistics
    logging.info(f"Sequence length: {sequences.shape[1]}, Feature dimension: {sequences.shape[2]}")
    logging.info(f"Label shape: {labels.shape}")
    
    return sequences, labels, scaler

# Evaluation metrics
def calculate_metrics(actuals, predictions):
    mse = np.mean(np.square(actuals - predictions))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    euclidean_distance = np.mean(np.sqrt(np.sum(np.square(actuals - predictions), axis=1)))
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Euclidean Distance": euclidean_distance
    }

# Training function with tqdm and TensorBoard
def train_model(model, train_loader, val_loader, optimizer, criterion, device, output_dir, epochs=50, writer=None):
    logging.info("Step 2: Starting model training.")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Progress bar for training steps
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Batch Loss": loss.item()})
        
        val_loss, metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            for metric, value in metrics.items():
                writer.add_scalar(f"Metrics/{metric}", value, epoch)
            writer.flush()

        logging.info(f"Epoch {epoch+1}/{epochs} Complete - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the model
    model_save_path = os.path.join(output_dir, f"model_epochs_{epochs}.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Step 2 Complete: Model saved to {model_save_path}")

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_actuals = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).cpu().numpy()  # Convert to NumPy
            actuals = y_batch.cpu().numpy()       # Convert to NumPy
            
            all_preds.append(preds)
            all_actuals.append(actuals)
            
            loss = criterion(torch.tensor(preds), torch.tensor(actuals))
            val_loss += loss.item()
    
    # Concatenate all predictions and actuals for metric calculation
    all_preds = np.vstack(all_preds)
    all_actuals = np.vstack(all_actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(all_actuals, all_preds)
    return val_loss / len(data_loader), metrics

# Prediction function
def predict(model, data_loader, device):
    logging.info("Step 3: Generating predictions.")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Predicting"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            predictions.extend(preds)
            actuals.extend(y_batch.numpy())
    logging.info("Step 3 Complete: Predictions generated.")
    return np.array(predictions), np.array(actuals)

# Main function to run the pipeline
def main(args):
    logging.info("Pipeline started.")
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Preprocess data
    sequences, labels, scaler = preprocess_data(args.data_file, args.window_size)
    
    # Split data into training and validation
    split_idx = int(0.8 * len(sequences))
    train_X, train_y = sequences[:split_idx], labels[:split_idx]
    val_X, val_y = sequences[split_idx:], labels[split_idx:]
    
    # Create DataLoaders
    train_loader = DataLoader(MobilityDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MobilityDataset(val_X, val_y), batch_size=args.batch_size, shuffle=False)
    
    # Select device
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() and args.gpus else "cpu")
    if torch.cuda.device_count() > 1 and args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        logging.info(f"Using GPUs: {args.gpus}")
    
    # Initialize model, optimizer, and loss function
    model = TimeSeriesTransformer(input_dim=2, model_dim=64, nhead=4, num_layers=2, output_dim=2)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Train and evaluate model
    train_model(model, train_loader, val_loader, optimizer, criterion, device, args.model_dir, epochs=args.epochs, writer=writer)
    
    writer.close()  # Close TensorBoard writer
    logging.info("Pipeline complete.")

# Argument parser for running the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Time-Series Transformer for mobility prediction.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the CSV file containing mobility data.")
    parser.add_argument("--window_size", type=int, default=10, help="Sliding window size for sequences.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--plot_dir", type=str, default="./plot", help="Directory to save plots.")
    parser.add_argument("--model_dir", type=str, default="./model", help="Directory to save models.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs.")
    parser.add_argument("--gpus", type=int, nargs='+', default=None, help="List of GPU IDs to use (e.g., 0 1 2).")
    args = parser.parse_args()
    
    main(args)
