import torch
import hydra

from omegaconf import DictConfig
from tqdm import tqdm
from data_utils import create_all_dataloaders
from model_utils import Weights, evaluate_model, create_model
from optim_utils import TrainLoopMirrorVRCS
from torch.utils.tensorboard import SummaryWriter
import torchvision



import torchvision.transforms as transforms

def cifar10_work():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Аугментации (RandAugment и Cutout можно добавить отдельно для ещё более высокого результата)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    # Загрузка CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    return {
        "train": trainloader,
        "test": testloader,
    }


def try_point():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model = MyModel(
    #    torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False).to(device)
    #)
    model = create_model()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #loaders = create_all_dataloaders(batch_size=500)
    loaders = cifar10_work()

    writer = SummaryWriter("runs/no_pretrain_true_my_data_work_true_model_try_point")

    for epoch in tqdm(range(200)):
        model.train()
        for X_batch, y_batch in loaders["train"]:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
        grad = Weights.from_grads(model)
        print(f"Grad Norm = {grad.norm()}")
        writer.add_scalar("Norm/Grad", grad.norm(), epoch)
        eval = evaluate_model(model, loaders["test"], device=device, criterion=criterion)
        writer.add_scalar("Accuracy/test", eval["accuracy"], epoch)
        writer.add_scalar("Loss/test", eval["loss"], epoch)
        print(f"Test: {eval}")





@hydra.main(config_path="config", config_name="config")
def start_point(cfg: DictConfig):
    print(f"cfg = {cfg}")
    #logger = SummaryWriter(f"runs/theta_{cfg.theta}_p_{cfg.p}_q_{cfg.q}_")
    loop = TrainLoopMirrorVRCS(cfg)
    loop.train()


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loaders = create_all_dataloaders(batch_size=500)

    writer = SummaryWriter("runs/no_pretrain_my_true_data_work_true_model")

    for epoch in tqdm(range(200)):
        model.train()
        for X_batch, y_batch in loaders["train"]:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
        grad = Weights.from_grads(model)
        print(f"Grad Norm = {grad.norm()}")
        writer.add_scalar("Norm/Grad", grad.norm(), epoch)
        eval = evaluate_model(model, loaders["test"], device=device, criterion=criterion)
        writer.add_scalar("Accuracy/test", eval["accuracy"], epoch)
        writer.add_scalar("Loss/test", eval["loss"], epoch)
        print(f"Test: {eval}")


if __name__ == "__main__":
    print("Hello mirror")
    #try_point()
    #test()
    start_point()





