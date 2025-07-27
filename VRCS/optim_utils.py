import torch
from tqdm import tqdm
from data_utils import create_all_dataloaders
from model_utils import forward_communication_gradient, Weights, evaluate_model, create_model
from hydra.utils import instantiate


class TrainLoopMirrorVRCS:
    def __init__(self, cfg):

        self.logger = instantiate(cfg.logger)
        self.log_acc_times = 0
        self.log_extragrad_full_com_times = 0
        self.log_extragrad_f_com_times = 0
        self.log_extragrad_g_com_times = 0
        self.log_argmin_loss_times = 0
        self.log_argmin_loss_grad_times = 0


        self.num_epochs=cfg.num_epochs_minimize
        self.iteration = cfg.iteration
        self.batch_size=cfg.batch_size
        self.p=cfg.p
        self.q=cfg.q
        self.theta=cfg.theta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = create_model()
        self.model.to(self.device)

        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())


        self.loaders = create_all_dataloaders(batch_size=self.batch_size)
        """
        loaders["server_f"]
        loaders["server_g"]
        loaders["clients_f"]
        loaders["clients_g"]
        loaders["individual_clients"]
        loaders["test"]
        """

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.grad_f = lambda model: forward_communication_gradient(model, self.loaders["clients_f"], criterion=self.criterion)["grad"]
        self.grad_g = lambda model: forward_communication_gradient(model, self.loaders["clients_g"], criterion=self.criterion)["grad"]
        self.grad_f1 = lambda model: forward_communication_gradient(model, self.loaders["server_f"], criterion=self.criterion)["grad"]
        self.grad_g1 = lambda model: forward_communication_gradient(model, self.loaders["server_g"], criterion=self.criterion)["grad"]
        self.grad_h1 = lambda model: (self.grad_f1(model) + self.grad_g1(model)) / 2
        self.grad_h = lambda model: (self.grad_f(model) + self.grad_g(model)) / 2

        torch.manual_seed(42)
        self.geom_dist = lambda q: int(torch.distributions.geometric.Geometric(q).sample())
        self.bern_dist = lambda p: int(torch.distributions.binomial.Binomial(total_count=1, probs=p).sample())


    def train(self):
        self.x_0 = Weights.from_model(self.model).clone()
        self.x_t = Weights.from_model(self.model).clone()
        for iteration in range(self.iteration):
            print(f"\nITERATION {iteration}\n")

            # Full communication + Argmin solve
            self.x_t.to_model(self.model)
            extragrad = self.grad_h(self.model) - self.grad_h1(self.model)
            self.x_t = self.minimize_A(extragrad=extragrad, model=self.model, x_t=self.x_t, optimizer=self.optimizer, theta=self.theta)
            self.x_0 = self.x_t.clone()

            self.log_accuracy_loss()
            self.log_extragrad_full_com(extragrad=extragrad)

            steps = self.geom_dist(self.q)
            print(f"{steps} local steps")
            self.vrcs_steps(steps=steps, p=self.p, theta=self.theta)



    def vrcs_steps(self, steps: int = 1, p: int = 0.0, theta: int = 0.0042):
        
        for step in range(steps):
            prob = self.bern_dist(p)

            if prob == 1.0:
                # f communication
                self.x_t.to_model(self.model)
                extragrad = p * (self.grad_f(self.model) - self.grad_f1(self.model))

                self.x_0.to_model(self.model)
                extragrad -= p * (self.grad_f(self.model) - self.grad_f1(self.model))
                self.log_extragrad_f_com(extragrad=extragrad)

            elif prob == 0.0:
                # g communication
                self.x_t.to_model(self.model)
                extragrad = (1 - p) * (self.grad_g(self.model) - self.grad_g1(self.model))

                self.x_0.to_model(self.model)
                extragrad -= (1 - p) * (self.grad_g(self.model) - self.grad_g1(self.model))
                self.log_extragrad_g_com(extragrad=extragrad)
            
            self.x_0.to_model(self.model)
            extragrad += self.grad_h(self.model) - self.grad_h1(self.model)
                
            self.x_t.to_model(self.model)
            self.x_t = self.minimize_A(extragrad=extragrad, model=self.model, x_t=self.x_t, optimizer=self.optimizer, theta=theta)

            self.print_metrics()
            self.log_accuracy_loss()

    def minimize_A(self, extragrad: Weights, model: torch.nn.Module, x_t: Weights, optimizer, theta):
        model.zero_grad()
        x_t = x_t.clone()
        x_t.freeze()
        x = Weights.from_model(model)

        for epoch in tqdm(range(self.num_epochs), desc="Argmin solve"):
            for inputs, targets in self.loaders["server"]:
                model.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model(inputs)

                third_part = self.criterion(output, targets)
                first_part = Weights.dot(extragrad, x)
                second_part = (x - x_t).norm()**2 / (2 * theta)
 
                loss = first_part + second_part + third_part
                loss.backward()
                optimizer.step()

                self.log_argmin_loss(loss.item())
                self.log_argmin_grad_norm(x.norm())
        return x.clone()

    def eval_model(self, dataloader):
        self.x_t.to_model(self.model)
        return evaluate_model(self.model, dataloader=dataloader, device=self.device, criterion=self.criterion)

    def print_metrics(self):
        load_1 = self.loaders["test"]
        load_2 = self.loaders["server"]
        print(f"Test: {self.eval_model(load_1)}", flush=True)
        print(f"Server data: {self.eval_model(load_2)}", flush=True)
    
    def log_accuracy_loss(self):
        self.log_acc_times += 1
        eval_test = self.eval_model(self.loaders["test"])
        eval_server = self.eval_model(self.loaders["server"])
        self.logger.add_scalar("accuracy/test", eval_test["accuracy"] , self.log_acc_times)
        self.logger.add_scalar("accuracy/server", eval_server["accuracy"] , self.log_acc_times)
        self.logger.add_scalar("loss/test", eval_test["loss"] , self.log_acc_times)
        self.logger.add_scalar("loss/server", eval_server["loss"] , self.log_acc_times)


    '''
        self.log_acc_times = 0
        self.log_extragrad_full_com_times = 0
        self.log_extragrad_f_com_times = 0
        self.log_extragrad_g_com_times = 0
        self.log_argmin_loss_times = 0
        self.log_argmin_loss_grad_times = 0
    '''

    def log_extragrad_full_com(self, extragrad):
        self.log_extragrad_full_com_times += 1
        norm = extragrad.norm()
        self.logger.add_scalar("norm/extragrad/full", norm , self.log_extragrad_full_com_times)

    def log_extragrad_f_com(self, extragrad):
        self.log_extragrad_f_com_times += 1
        norm = extragrad.norm()
        self.logger.add_scalar("norm/extragrad/f", norm , self.log_extragrad_f_com_times)

    def log_extragrad_g_com(self, extragrad):
        self.log_extragrad_g_com_times += 1
        norm = extragrad.norm()
        self.logger.add_scalar("norm/extragrad/g", norm , self.log_extragrad_g_com_times)
    
    def log_argmin_loss(self, loss):
        self.log_argmin_loss_times += 1
        self.logger.add_scalar("loss/argmin", loss , self.log_argmin_loss_times)


    def log_argmin_grad_norm(self, norm):
        self.log_argmin_loss_grad_times += 1
        self.logger.add_scalar("norm/argmin/loss_grad", norm , self.log_argmin_loss_grad_times)