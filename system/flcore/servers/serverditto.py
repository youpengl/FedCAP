from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
import copy
from utils.byzantine import *
import time

class Ditto(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientDitto)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

    def train(self):
        client_times = []
        server_times = []
        for i in range(self.global_rounds+1):
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\n-Evaluate global models for training clients")
                self.evaluate()

                self.resume_client_model()

            if i == self.global_rounds:
                break
            
            start = time.time()    
            for client in self.selected_clients:
                client.dtrain()
            client_times.append(time.time()-start)
            
            if i%self.eval_gap == 0:
                print("\n-Evaluate personalized models for training clients ")
                self.evaluate_personalized()
  
            self.receive_models()

            start = time.time()
            self.aggregate_parameters()
            server_times.append(time.time()-start)

        print(f"\n-------------Final Report-------------")
        print("\nFinal Testing Accuracy: {:.2f}%".format(max(self.rs_test_acc_g[-1]*100, self.rs_test_acc_p[-1]*100)))
        print(f"\nClient computation time cost: {sum(client_times)}s.")
        print(f"\nServer computation time cost: {sum(server_times)}s.")

    def receive_models(self):
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_updates = []

        tot_samples = 0

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_updates.append([c_param.data - s_param.data for c_param, s_param in zip(client.model.parameters(), self.global_model.parameters())])
    
        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(None, self.uploaded_updates, malicious_ids)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        self.global_update = copy.deepcopy(self.uploaded_updates[0])
        for param in self.global_update:
            param.data.zero_()
            
        for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
            self.add_parameters(w, client_update)

        for model_param, update_param in zip(self.global_model.parameters(), self.global_update):
            model_param.data += update_param.data.clone()

    def add_parameters(self, w, client_update):
        for server_param, client_param in zip(self.global_update, client_update):
            server_param.data += client_param.data.clone() * w