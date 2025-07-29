
# Federated Learning

## Работа с весами

Для того чтобы не заморачиваться с операциями над весами модели или градиентами, создадим класс, который содержит в себе словарь весов моделей и позволяет удобно работать с ними

### Интерфейс

#### Инициализация и обратная инициализация

```python
model: torch.nn.Module

x = Weigths.from_model(model)
grad_x = Weights.from_grads(model)
```

- Веса `x` указывают на веса модели `model`, это позволяет сохранять граф вычислений при применении операций к весам для оптимизации
- Градиенты `grad_x` отвязаны от `model` и при дальнейшем использовании `loss.backward()` или `model.zero_grad()` `optimizer.zero_grad` градиент `grad_x` не будет изменяться

```python
x: Weights
model: torch.nn.Model

x.to_model(model)
```

- Теперь модель `model` будет иметь веса `x`, но при этом граф вычислений не передается, !только данные!
- Так можно делать с любым объектом класса, даже с градиентами
#### Отвязка от графа вычислений
```python
model: torch.nn.Module
x: Weights

x_t = Weight.from_model(model)
x_t.clone() <-- отвязка от параметров model
x_t.freeze() <-- заморозка весов
x = Weights.from_model(model)

loss = (x - x_t).norm()**2 / 2
loss.backward()
```

- В таком варинте веса `x` указывают на параметры `model` и все операции влияют на градиент
- `x_t` очевидно указывает на совсем другие параметры, поэтому дальнейшие действия с ними не влияет на `x` и `model`
- Заморозка позволяет избавиться от лишних вычислений
#### Операции над весами


```python
x, y: weigths

x = x + y
x = x - y
x = Weights.dot(x, y)

x = 2 * x
x = x / 2
```

- Объекты этого класса можно складывать, вычитать, умножать и делить на скаляр, скалярно умножать
- Само собой операции учитываются в графе вычислений

```python
x, x_t, grad_x: Weights
alpha: int

optimizer = Adam_or_smtng_else(model.parameters())
model.zero_grad()
x_t = Weights.from_model(model).clone() <- не привязаны к графу вычислений
x = Weight.from_model(model) <- Привязаны к графу вычислений

# Realization 1 step of SGD in argmin theory

output = model(input)
loss = criterion(target, output)
loss.backward()
grad_x = Weights.from_grads(model)
model.zero_grad()

loss = Weights.dot(alpha * grad_x, x) + (x - x_t).norm(p=2)
loss.backward()
optimizer.step()
```



## Start

Create and use virtual env
```bash
python -m venv env
source env/bin/activate
```

```bash
pip install torch
pip install hydra_core
pip install tensorboard
pip install tqdm
pip install torchvision
```

### Launch
```bash
CUDA_VISIBLE_DEVICES=0 nohup python \
federated_methods/VRCS_new/start.py \
theta=0.01 \
p=0.5 \
q=0.5 \
> outputs/my_code_theta_$(theta)_p_$(p)_q_$(q)_$(postfix).txt &
```
