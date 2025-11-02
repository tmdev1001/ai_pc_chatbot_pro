"""
Model training script for chatbot intent classification.
Trains a neural network to classify user queries into intents.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from training_data import get_training_data


class ChatDataset(Dataset):
    """Dataset class for chatbot training data."""
    
    def __init__(self, data):
        self.samples = data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


def create_model():
    """Create and return the neural network model."""
    return nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )


def setup_plotting():
    """Initialize and return the matplotlib figure and axes for real-time plotting."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training & Validation Progress - Real-time Updates', fontsize=16, fontweight='bold')
    
    # Initialize plots for both training and validation
    train_loss_line, = ax1.plot([], [], 'b-', linewidth=2, label='Train Loss')
    val_loss_line, = ax1.plot([], [], 'r--', linewidth=2, label='Val Loss')
    train_acc_line, = ax2.plot([], [], 'g-', linewidth=2, label='Train Acc')
    val_acc_line, = ax2.plot([], [], 'orange', linewidth=2, linestyle='--', label='Val Acc')
    
    # Configure loss plot
    ax1.set_title('Loss (Training & Validation)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Configure accuracy plot
    ax2.set_title('Accuracy (Training & Validation)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line


def update_plot(ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line,
                train_losses, val_losses, train_accuracies, val_accuracies):
    """Update the plots with new training and validation data."""
    epochs_list = list(range(1, len(train_losses) + 1))
    train_loss_line.set_data(epochs_list, train_losses)
    val_loss_line.set_data(epochs_list, val_losses)
    train_acc_line.set_data(epochs_list, train_accuracies)
    val_acc_line.set_data(epochs_list, val_accuracies)
    
    # Auto-scale axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax2.set_ylim([0, 105])  # Keep accuracy between 0-105%
    
    # Update display
    plt.draw()
    plt.pause(0.01)


def train_epoch(model, loader, criterion, optimizer):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for x, y in loader:
        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total_samples += y.size(0)
        correct_predictions += (predicted == y).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = epoch_loss / len(loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    """Evaluate the model on validation/test data and return loss and accuracy."""
    model.eval()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            # Forward pass
            pred = model(x)
            loss = criterion(pred, y)
            
            # Calculate metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = epoch_loss / len(loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    # Configuration
    NUM_EPOCHS = 1000
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    MODEL_PATH = "intent_model.pt"
    GRAPH_PATH = "training_progress.png"
    
    # Load training data
    print("Loading training data...")
    data = get_training_data()
    print(f"Loaded {len(data)} training examples")
    
    # Split data into training (80%) and validation (20%)
    dataset = ChatDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize tracking lists and plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    fig, ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line = setup_plotting()
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update graphs in real-time
        update_plot(ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line,
                   train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Display progress every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch [{epoch+1:4d}/{NUM_EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    print("-" * 60)
    print("Training completed!")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Final update and save graph
    plt.savefig(GRAPH_PATH, dpi=300, bbox_inches='tight')
    print(f"Training progress graphs saved to {GRAPH_PATH}")
    
    # Keep window open (turn off interactive mode to keep plot visible)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
