import torch
import torch.nn.functional as F

# Logits (raw scores) for a batch of 2 samples and 3 classes
logits = torch.tensor([[1.0, 2.0, 3.0], [14.0, 2.0, 3.0]])  # Shape: [2, 3]
labels = torch.tensor([2, 1])  # True labels for the batch, Shape: [2]

# Using PyTorch's CrossEntropyLoss
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

# Calculating manually for clarity
log_softmax_logits = F.log_softmax(logits, dim=1)
print(f"log_softmax_logits:\n{log_softmax_logits}")

# For the first sample (true class 2)
loss_1 = -log_softmax_logits[0, 2]  # -(-0.4076) = 0.4076
# For the second sample (true class 1)
loss_2 = -log_softmax_logits[1, 1]  # -(-1.4076) = 1.4076

# Average loss
average_loss = (loss_1 + loss_2) / 2

print(f"Loss from CrossEntropyLoss: {loss.item()}")
print(f"Manually calculated loss: {average_loss.item()}")