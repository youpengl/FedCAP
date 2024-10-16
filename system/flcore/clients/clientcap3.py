import copy
import torch
import numpy as np
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from sklearn.preprocessing import label_binarize

class clientCAP3(Client):
    def __init__(self, args, id, malicious, **kwargs):
        super().__init__(args, id, malicious, **kwargs)

        self.lamda = args.lamda
        self.c_per_model = copy.deepcopy(self.model) # received model per round
        self.itentity_as_malicious = False
        
        self.model_per = copy.deepcopy(self.model) # personalized model
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, lamda=self.lamda)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()

        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

    def test_metrics_personalized(self):

        testloaderfull = self.load_test_data()

        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        return test_acc, test_num

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.lamda * torch.norm(pm-gm, p=2)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    def get_update(self, global_model):

        trainloader = self.load_train_data()

        model = copy.deepcopy(self.model)
        self.set_parameters(global_model)
        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        model_update = [c_param.data - s_param.data for c_param, s_param in zip(self.model.parameters(), global_model.parameters())]
    
        self.set_parameters(model)

        return model_update

