import copy
from typing import List
import torch
from torchvision.models import resnet18


class Weights:
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data

    @classmethod
    def from_model(cls, model: torch.nn.Module):
        return cls({name: param for name, param in model.named_parameters()})

    @classmethod
    def from_grads(cls, model: torch.nn.Module):
        return cls({
            name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            for name, param in model.named_parameters()
        })
    
    def to_model(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.data:
                    param.data.copy_(self.data[name])
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Weights(
                {k: self.data[k] + other.data[k] for k in self.data}
            )
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return Weights(
                {k: self.data[k] - other.data[k] for k in self.data}
            )
        else:
            return NotImplemented


    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Weights({k: v / other for k, v in self.data.items()})
        else:
            return NotImplemented

    def to(self, device):
        return Weights({
            k: v.to(device) for k, v in self.data.items()
        })
        
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return NotImplemented

        # other == scalar
        return Weights(
            {k: v * other for k, v in self.data.items()}
        )
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def dot(cls, obj1, obj2):
        return sum(
            torch.sum(obj1.data[k] * obj2.data[k])
            for k in obj1.data
        )

    def norm(self, p: float = 2.0):
        return torch.norm(
            torch.cat([v.flatten() for v in self.data.values()]),
            p=p
        )

    def freeze(self):
        for v in self.data.values():
            v.requires_grad = False


    def clone(self):
        return Weights({k: v.detach().clone() for k, v in self.data.items()})


def create_model():
    model = resnet18(num_classes=10)
    # Убираем downsampling в начале
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model

def forward_communication_gradient(model, dataloader, criterion, device="cuda") -> Weights:
    model.train()
    model.zero_grad()
    print("Computing data ...")
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    return {
        "grad": Weights.from_grads(model),
        "loss": loss
    }

@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()  # перевод в режим оценки
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        if criterion is not None:
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # домножаем на размер батча

        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples if criterion is not None else None

    return {
        "accuracy": accuracy,
        "loss": avg_loss
    }