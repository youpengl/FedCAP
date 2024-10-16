import copy
from flcore.clients.clientbase import Client

class clientCAP1(Client):
    def __init__(self, args, id, malicious, **kwargs):
        super().__init__(args, id, malicious, **kwargs)

        self.lamda = args.lamda
        self.c_per_model = copy.deepcopy(self.model) # received model per round
        self.itentity_as_malicious = False
    
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

