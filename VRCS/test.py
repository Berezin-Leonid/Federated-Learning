from data_utils import create_all_dataloaders
from model_utils import *


device = "cuda" if torch.cuda.is_available() else "cpu"

#test="DATA"
test="WEIGHTS"

def weights_test_1():
    # Testing dot grad here
    model = MyModel(
        torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False).to(device)
    )
    model.to(device)


    model.zero_grad()
    x = Weights.from_model(model)
    grad_start = Weights.from_grads(model)
    prod = Weights.dot(x, x)
    prod.backward()
    prod_grad = Weights.from_grads(model)
    print(f"Норма начала градиента = {grad_start.norm()}")
    print(f"Норма весов = {x.norm()}")
    print(f"Норма градиента после dot= {prod_grad.norm()}")

def weights_test_2():
    # Testing dot grad here
    model = MyModel(
        torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False).to(device)
    )
    model.to(device)


    model.zero_grad()
    x = Weights.from_model(model)
    grad_start = Weights.from_grads(model)
    final = Weights.norm(x)
    final.backward()
    prod_grad = Weights.from_grads(model)
    print(f"Норма начала градиента = {grad_start.norm()}")
    print(f"Норма весов = {x.norm()}")
    print(f"Норма градиента после нормы= {prod_grad.norm()}")

def weights_test_3():
    # Testing dot grad here
    model = MyModel(
        torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False).to(device)
    )
    model.to(device)


    model.zero_grad()
    x = Weights.from_model(model)
    grad_start = Weights.from_grads(model)
    prod = Weights.dot((1/226) * x, x)
    norm = Weights.norm(x)
    final = prod + norm
    final.backward()
    prod_grad = Weights.from_grads(model)
    print(f"Норма начала градиента = {grad_start.norm()}")
    print(f"Норма весов = {x.norm()}")
    print(f"Норма градиента после нормы= {prod_grad.norm()}")

if test=="DATA":
    loaders = create_all_dataloaders()
    print(loaders["server_f"])



if test=="WEIGHTS":
    print(f"======= Test 1: =======\n\n")
    weights_test_1()
    print(f"======= ======= =======\n\n")
    print(f"======= Test 2: =======\n\n")
    weights_test_2()
    print(f"======= ======= =======\n\n")

    print(f"======= Test 3: =======\n\n")
    weights_test_3()
    print(f"======= ======= =======\n\n")
