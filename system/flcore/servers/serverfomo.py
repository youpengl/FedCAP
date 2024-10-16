import torch
import copy
import numpy as np
from flcore.clients.clientfomo import clientFomo
from flcore.servers.serverbase import Server
from utils.byzantine import *
import time

class FedFomo(Server):
    def __init__(self, args):
        super().__init__(args)

        self.val_ratio = 0.2
        self.set_clients(args, clientFomo)

        self.P = torch.diag(torch.ones(self.num_training_clients, device=self.device))
        self.uploaded_ids = []
        self.uploaded_models = []
        self.O = min(args.O, self.join_clients)
        self.server_times = []
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

    def set_clients(self, args, clientObj):

        if self.attack_type == 'B':
            self.malicious_ids = []
            self.attack_ratio = 0.0
        else:
            self.malicious_ids = np.sort(np.random.choice(np.arange(self.num_clients), int(self.num_clients * self.attack_ratio), replace=False))
            
        for i in range(self.num_clients):
                    
            client = clientObj(args, 
                            id=i, 
                            malicious=True if i in self.malicious_ids else False)
            
            client.weight_vector = torch.zeros(self.num_clients, device=self.device)

            self.clients.append(client)
        
        self.training_clients = self.clients
        self.training_clients_ids = np.arange(self.num_clients)

        print('Malicious Clients: {}'.format(list(self.malicious_ids)))
    
    def train(self):
        client_times = []
        for i in range(self.global_rounds+1):
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models()

            # fomo
            for client in self.training_clients:

                if client not in self.selected_clients:
                    client.resume_model = copy.deepcopy(client.model)

                if client in self.selected_clients:
                    client.clone_model(client.model, client.old_model)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\n-Evaluate received models for training clients")
                self.evaluate()

                self.resume_client_model()

            if i == self.global_rounds:
                break
            
            start = time.time() 
            for client in self.selected_clients:
                client.train()
            client_times.append(time.time()-start)
            
            if i%self.eval_gap == 0:
                print("\n-Evaluate fine-tuned models for training clients")
                self.evaluate_personalized()

            self.receive_models()

        print(f"\n-------------Final Report-------------")
        print("\nFinal Testing Accuracy: {:.2f}%".format(max(self.rs_test_acc_g[-1]*100, self.rs_test_acc_p[-1]*100)))
        print(f"\nClient computation time cost: {sum(client_times)}s.")
        print(f"\nServer computation time cost: {sum(self.server_times)}s.")

    def send_models(self):

        for client in self.training_clients:
            possible_ids = [u_id for u_id in self.uploaded_ids]
            O_ = len(possible_ids) if len(possible_ids) < self.O else self.O
            indices = np.array(possible_ids)[torch.topk(self.P[client.id][possible_ids], O_).indices.tolist()]
            send_ids = []
            send_models = []
            for i in indices:
                send_ids.append(i)
                send_models.append(self.uploaded_models[self.uploaded_ids.index(i)])

            client.receive_models(send_ids, send_models)

    def receive_models(self):

        self.uploaded_ids = []
        self.uploaded_models = []
        model_updates = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            model_updates.append([new_param.data - old_param.data for new_param, old_param in zip(client.model.parameters(), client.old_model.parameters())])
            self.P[client.id] += client.weight_vector

        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            model_updates = eval(self.attack_type)(None, model_updates, malicious_ids)
        
        start = time.time() 
        for i, client in enumerate(self.selected_clients):

            model= copy.deepcopy(client.old_model)

            for j, param in enumerate(model.parameters()):
                param.data += model_updates[i][j].data.clone()

            self.uploaded_models.append(model)
        self.server_times.append(time.time()-start)