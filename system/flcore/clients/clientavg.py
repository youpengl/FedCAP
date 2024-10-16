from flcore.clients.clientbase import Client

class clientAVG(Client):
    def __init__(self, args, id, malicious, **kwargs):
        super().__init__(args, id, malicious, **kwargs)

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
