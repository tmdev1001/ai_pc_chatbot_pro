import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ChatDataset(Dataset):
    def __init__(self, data):
        self.samples = data
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y)

# Fake embedding (in real case use BERT or SentenceTransformer)
def simple_embed(text):
    return [len(text), text.count('?'), text.count('')] #trivial feature

data = [
    (simple_embed("Which app I used most?"), 0), # 0=most_used
    (simple_embed("show usage time?"), 1), # 1=usage_summary
    # Intent 0: Most Used App variations (50 examples)
    (simple_embed("What's my most used app?"), 0),
    (simple_embed("Which application did I use the most?"), 0),
    (simple_embed("Show me the app I used most"), 0),
    (simple_embed("What app have I been using the most?"), 0),
    (simple_embed("Which program is my top used app?"), 0),
    (simple_embed("What's the most frequently used app?"), 0),
    (simple_embed("Tell me my most used application"), 0),
    (simple_embed("Which app tops my usage list?"), 0),
    (simple_embed("What application have I opened most?"), 0),
    (simple_embed("Show the app with highest usage"), 0),
    (simple_embed("Which app do I use most often?"), 0),
    (simple_embed("What's my top app?"), 0),
    (simple_embed("Which software did I use most?"), 0),
    (simple_embed("What app is used the most by me?"), 0),
    (simple_embed("Display my most used app"), 0),
    (simple_embed("Which application is most popular on my PC?"), 0),
    (simple_embed("What's the app I launch most?"), 0),
    (simple_embed("Show top app by usage"), 0),
    (simple_embed("Which app has the most usage time?"), 0),
    (simple_embed("What application appears most in my usage?"), 0),
    (simple_embed("Tell me which app I use most"), 0),
    (simple_embed("What's the number one app I use?"), 0),
    (simple_embed("Which program do I open most?"), 0),
    (simple_embed("Show me the top used application"), 0),
    (simple_embed("What app ranks first in my usage?"), 0),
    (simple_embed("Which app is my favorite based on usage?"), 0),
    (simple_embed("What application do I use most frequently?"), 0),
    (simple_embed("Show my most used software"), 0),
    (simple_embed("Which app have I been spending most time on?"), 0),
    (simple_embed("What's my primary app?"), 0),
    (simple_embed("Which application tops the chart?"), 0),
    (simple_embed("What app did I use most today?"), 0),
    (simple_embed("Show the most opened app"), 0),
    (simple_embed("Which program is my go-to app?"), 0),
    (simple_embed("What application leads in usage?"), 0),
    (simple_embed("Tell me my top app"), 0),
    (simple_embed("Which app comes first in usage?"), 0),
    (simple_embed("What's the app with maximum usage?"), 0),
    (simple_embed("Show app that I use most"), 0),
    (simple_embed("Which application is number one?"), 0),
    (simple_embed("What app dominates my usage?"), 0),
    (simple_embed("Display the most used application"), 0),
    (simple_embed("Which program have I used most?"), 0),
    (simple_embed("What's my highest usage app?"), 0),
    (simple_embed("Show me top application"), 0),
    (simple_embed("show app mosed used "), 0),
    (simple_embed("Which app is used most frequently?"), 0),
    (simple_embed("What application has the highest usage?"), 0),
    (simple_embed("Tell me the app I use most"), 0),
    (simple_embed("Which software tops my usage?"), 0),
    (simple_embed("What app do I access most?"), 0),
    (simple_embed("Show the application I use most"), 0),
    # Intent 1: Usage Summary variations (50 examples)
    (simple_embed("How much time did I spend?"), 1),
    (simple_embed("What's my total usage time?"), 1),
    (simple_embed("Show me usage statistics"), 1),
    (simple_embed("Display my usage summary"), 1),
    (simple_embed("What's the total time I used apps?"), 1),
    (simple_embed("How long have I been using apps?"), 1),
    (simple_embed("Show total active time"), 1),
    (simple_embed("What's my overall usage?"), 1),
    (simple_embed("Give me a usage report"), 1),
    (simple_embed("How many minutes did I use apps?"), 1),
    (simple_embed("Show usage statistics"), 1),
    (simple_embed("What's the summary of my usage?"), 1),
    (simple_embed("Display total time spent"), 1),
    (simple_embed("How much time total?"), 1),
    (simple_embed("Show me my usage data"), 1),
    (simple_embed("What's my usage overview?"), 1),
    (simple_embed("Tell me my total usage time"), 1),
    (simple_embed("How long have I been active?"), 1),
    (simple_embed("Show total app usage time"), 1),
    (simple_embed("What's the cumulative usage?"), 1),
    (simple_embed("Display usage breakdown"), 1),
    (simple_embed("How much time in total?"), 1),
    (simple_embed("Show usage analytics"), 1),
    (simple_embed("What's my time spent summary?"), 1),
    (simple_embed("Give me usage details"), 1),
    (simple_embed("How many hours did I use?"), 1),
    (simple_embed("Show total time"), 1),
    (simple_embed("What's my overall activity time?"), 1),
    (simple_embed("Display usage information"), 1),
    (simple_embed("How much total usage?"), 1),
    (simple_embed("Show me usage totals"), 1),
    (simple_embed("What's the total active duration?"), 1),
    (simple_embed("Tell me total usage"), 1),
    (simple_embed("How long total time?"), 1),
    (simple_embed("Show usage summary"), 1),
    (simple_embed("What's my cumulative time?"), 1),
    (simple_embed("Display total activity"), 1),
    (simple_embed("How much time altogether?"), 1),
    (simple_embed("Show overall usage"), 1),
    (simple_embed("What's the total duration?"), 1),
    (simple_embed("Give me total usage stats"), 1),
    (simple_embed("How many minutes total?"), 1),
    (simple_embed("Show usage report"), 1),
    (simple_embed("What's my total activity?"), 1),
    (simple_embed("Display time summary"), 1),
    (simple_embed("How much usage time total?"), 1),
    (simple_embed("Show me the totals"), 1),
    (simple_embed("What's the usage breakdown?"), 1),
    (simple_embed("Tell me usage time summary"), 1),
    (simple_embed("How long total?"), 1),
    (simple_embed("Show total usage statistics"), 1),
]

loader = DataLoader(ChatDataset(data), batch_size=16, shuffle=True)
model = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Lists to store training metrics
train_losses = []
train_accuracies = []

# Enable interactive mode for real-time plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training Progress - Real-time Updates', fontsize=16, fontweight='bold')

# Initialize plots
loss_line, = ax1.plot([], [], 'b-', linewidth=2)
acc_line, = ax2.plot([], [], 'g-', linewidth=2)

ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Starting training...")
print("-" * 60)

for epoch in range(100):
    # Training phase
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
    
    # Calculate average loss and accuracy for this epoch
    avg_loss = epoch_loss / len(loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    # Update graphs in real-time
    epochs_list = list(range(1, len(train_losses) + 1))
    loss_line.set_data(epochs_list, train_losses)
    acc_line.set_data(epochs_list, train_accuracies)
    
    # Auto-scale axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax2.set_ylim([0, 105])  # Keep accuracy between 0-105%
    
    # Update display
    plt.draw()
    plt.pause(0.01)  # Small pause to update the display
    
    # Display progress every 10 epochs or on the last epoch
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == 99:
        print(f"Epoch [{epoch+1:3d}/100] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

print("-" * 60)
print("Training completed!")

# Save the model
torch.save(model.state_dict(), "intent_model.pt")
print("Model saved to intent_model.pt")

# Final update and save
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
print("Training progress graphs saved to training_progress.png")

# Keep window open (turn off interactive mode to keep plot visible)
plt.ioff()
plt.show()