import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
]

loader = DataLoader(ChatDataset(data), batch_size=2, shuffle=True)
model = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    for x,y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad();loss.backward();optimizer.step()

torch.save(model.state_dict(), "intent_model.pt")