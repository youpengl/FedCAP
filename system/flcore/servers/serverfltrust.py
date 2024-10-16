from flcore.clients.clientfltrust import clientFLTrust
from flcore.servers.serverbase import Server
import copy
from utils.byzantine import *
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FLTrust(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientFLTrust)
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

    def train(self):
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
                
            for client in self.selected_clients:
                client.train()

            if i%self.eval_gap == 0:
                print("\n-Evaluate fine-tuned models for training clients")
                self.evaluate_personalized()

            self.receive_models()
            self.aggregate_parameters()

        print(f"\n-------------Final Report-------------")
        print("\nFinal Testing Accuracy: {:.2f}%".format(max(self.rs_test_acc_g[-1]*100, self.rs_test_acc_p[-1]*100)))
        self.save_global_model()


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
            self.malicious_true_ids_current_round = [c_id for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(None, self.uploaded_updates, malicious_ids)

        client_updates = [torch.cat([p.data.reshape(1, -1) for p in u], dim=1) for u in self.uploaded_updates]
        server_update = torch.cat([p.data.reshape(1, -1) for p in self.get_update()], dim=1)

        similarity = []
        selected_true_idx = []
        for i, client_update in enumerate(client_updates):
            simi = F.cosine_similarity(client_update, server_update)[0].item()
            if simi < 0:
                simi = 0
            else:
                selected_true_idx.append(self.uploaded_ids[i])
            similarity.append(simi)
            
        self.uploaded_weights = [simi/(sum(similarity)+1e-9) for simi in similarity]

        norm_server_update = torch.norm(server_update).item()

        for i in range(len(client_updates)):
            norm_client_update = torch.norm(client_updates[i]).item()
            for j in range(len(client_updates[i])):
                try:
                    self.uploaded_updates[i][j] = norm_server_update / norm_client_update * self.uploaded_updates[i][j]
                except Exception:
                    self.uploaded_updates[i][j] = self.uploaded_updates[i][j]

            

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

    def load_server_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        with np.load('{}/server/server.npz'.format(self.data_path, self.dataset), allow_pickle=True) as f:
            server_data = f['server_data'][()]
        server_x = torch.Tensor(server_data['x']).type(torch.float32)
        server_y = torch.Tensor(server_data['y']).type(torch.int64)
        server_data = [[x, y] for x, y in zip(server_x, server_y)]
        return DataLoader(server_data, self.batch_size, drop_last=True, shuffle=False)
    
    def get_update(self):

        trainloader = self.load_server_data()
        
        model = copy.deepcopy(self.global_model)
        self.global_model.train()

        max_local_steps = self.local_steps
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                loss = self.clients[0].loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        model_update = [c_param.data - s_param.data for c_param, s_param in zip(self.global_model.parameters(), model.parameters())]
        
        for old_param, new_param in zip(self.global_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

        return model_update