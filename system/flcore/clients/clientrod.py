import copy
import torch
import numpy as np
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize


class clientROD(Client):
    def __init__(self, args, id, malicious, **kwargs):
        super().__init__(args, id, malicious, **kwargs)
                
        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt_head, 
            gamma=args.learning_rate_decay_gamma
        )

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1


    def train(self):
        trainloader = self.load_train_data()
        
        self.model.train()
        self.head.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                out_p = self.head(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_head.step()

    def test_metrics(self, model=None):
        testloader = self.load_test_data()

        if model == None:
            model = self.model

        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        return test_acc, test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = balanced_softmax_loss(y, output, self.sample_per_class)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics_personalized(self, model=None):
        testloader = self.load_test_data()
        
        if model == None:
            model = self.model
        
        model.eval()
        self.head.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        return test_acc, test_num

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        
        self.model.eval()
        self.head.eval()
        
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    
# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
