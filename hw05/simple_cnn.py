import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*14*14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, 16*14*14)
        return self.fc(x)

# 数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False
)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img), label)
        loss.backward()
        optimizer.step()

# 测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        correct += (model(img).argmax(1) == label).sum().item()
        total += label.size(0)
print(f"极简CNN测试准确率: {100*correct/total:.2f}%")