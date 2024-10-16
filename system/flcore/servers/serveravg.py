from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import copy
from utils.byzantine import *
from utils.defense import agg_func
import time

class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientAVG)
        self.aggregation = args.aggregation
        self.norm = []
        self.gas = args.gas
        self.gas_p = args.gas_p
        self.bucket = args.bucket
        self.bucket_s = args.bucket_s
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

            # client computation time
            start = time.time()    
            for client in self.selected_clients:
                client.train()
            client_times.append(time.time()-start)

            if i%self.eval_gap == 0:
                print("\n-Evaluate fine-tuned models for training clients")
                self.evaluate_personalized()

            self.receive_models()

            #  server computation time
            start = time.time()    
            self.aggregate_parameters()
            server_times.append(time.time()-start)
            
        print(f"\n-------------Final Report-------------")
        print("\nFedAvg: Final Testing Accuracy: {:.2f}%".format(self.rs_test_acc_g[-1]*100))
        print("\nFedAvg-FT: Final Testing Accuracy: {:.2f}%".format(self.rs_test_acc_p[-1]*100))
        print(f"\nClient computation time cost: {sum(client_times)}s.")
        print(f"\nServer computation time cost: {sum(server_times)}s.")
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
        
        # fig.3
        if self.attack_type in ['A3', 'A4']:
            self.avg_benign_norm = 0
            self.avg_malicious_norm = 0
            for i, true_id in enumerate(self.uploaded_ids):
                update = self.uploaded_updates[i]
                u = torch.cat([p.data.reshape(1, -1) for p in update], dim=1)
                norm = torch.norm(u).item()
                if true_id in self.malicious_ids:
                    self.avg_malicious_norm += norm
                else:
                    self.avg_benign_norm += norm
            
            self.avg_malicious_norm /= len(self.malicious_true_ids_current_round)
            self.avg_benign_norm /= len(self.uploaded_ids) - len(self.malicious_true_ids_current_round)

    def aggregate_parameters(self):
        
        self.global_update = copy.deepcopy(self.uploaded_updates[0])
        for param in self.global_update:
            param.data.zero_()
        
        if self.gas:
            # flatten
            flat_updates = []
            for update in self.uploaded_updates:
                flat_params = []
                shape_tuples = []
                for param in update:
                    flat_params.append(param.view(-1))
                    shape_tuples.append(param.shape)
                flat_updates.append(torch.cat(flat_params))
            flat_updates = torch.stack(flat_updates)
            
            # splitting
            d = flat_updates.shape[1]
            shuffled_dims = torch.randperm(d).to(flat_updates.device)
            p = self.gas_p
            partition = torch.chunk(shuffled_dims, chunks=p)
            groups = [flat_updates[:, partition_i] for partition_i in partition]

            # aggregate sub-vector and calculate identification score
            base_agg = agg_func[self.aggregation]
            n_cl = len(flat_updates)
            n_sel = len(self.selected_clients)
            n_sel_byz = int(n_sel*self.attack_ratio)

            identification_scores = torch.zeros(n_cl)
            for group in groups:
                uploaded_updates, uploaded_weights = base_agg(group, self.uploaded_weights, n_cl, n_sel, n_sel_byz)
                
                group_agg = copy.deepcopy(group[0])
                for param in group_agg:
                    param.data.zero_()
                for client_param, w in zip(uploaded_updates, uploaded_weights):
                    group_agg.data += client_param.data.clone() * w

                group_scores = (group - group_agg).square().sum(dim=-1).sqrt().cpu()
                identification_scores += group_scores
            _, cand_idxs = identification_scores.topk(k=n_cl - n_sel_byz, largest=False)
            n_agg_byz = sum([self.clients[self.uploaded_ids[i]].malicious for i in cand_idxs.tolist()])
            print(f"Aggregated byzantine / Expected selected byzantine: {n_agg_byz} / {n_sel_byz}")

            # vector aggregation
            flat_agg_update = flat_updates[cand_idxs].mean(dim=0)

            split_size = [t.numel() for t in shape_tuples]
            split_tensors = torch.split(flat_agg_update, split_size)
            self.global_update = [split_tensor.view(shape) for shape, split_tensor in zip(shape_tuples, split_tensors)]

        elif self.bucket:
            indices = list(range(len(self.uploaded_updates)))
            np.random.seed(self.seed)
            np.random.shuffle(indices)

            T = int(np.ceil(len(indices) / self.bucket_s))

            reshuffled_inputs = []
            for t in range(T):
                indices_slice = indices[t * self.bucket_s : (t + 1) * self.bucket_s]
                g_bar = copy.deepcopy(self.uploaded_updates[0])
                for param in g_bar:
                    param.data.zero_()
                bucket_updates = [uu for i, uu in enumerate(self.uploaded_updates) if i in indices_slice]
                bucket_weights = [1 / len(indices_slice) for i in range(len(indices_slice))]
                for bucket_update, w in zip(bucket_updates, bucket_weights):
                    for g_param, update_param in zip(g_bar, bucket_update):
                        g_param.data += update_param.data.clone() * w

                reshuffled_inputs.append(g_bar)
            
            reshuffled_weights = [1 / len(reshuffled_inputs) for i in range(len(reshuffled_inputs))]
            n_cl = n_sel = len(reshuffled_inputs)
            n_sel_byz = int(n_sel*self.attack_ratio)
            base_agg = agg_func[self.aggregation]
            self.uploaded_updates, self.uploaded_weights = base_agg(reshuffled_inputs, reshuffled_weights, n_cl, n_sel, n_sel_byz)
            for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
                self.add_parameters(w, client_update)

        else:
            if self.aggregation == 'mean':
                self.uploaded_weights = [i/sum(self.uploaded_weights) for i in self.uploaded_weights]           
            else:
                n_cl = len(self.uploaded_updates)
                n_sel = len(self.selected_clients)
                n_sel_byz = int(n_sel*self.attack_ratio)
                base_agg = agg_func[self.aggregation]
                self.uploaded_updates, self.uploaded_weights = base_agg(self.uploaded_updates, self.uploaded_weights, n_cl, n_sel, n_sel_byz)

            for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
                self.add_parameters(w, client_update)

        #fig.3
        if self.attack_type in ['A3', 'A4']:
            u = torch.cat([p.data.reshape(1, -1) for p in self.global_update], dim=1)
            avg_norm = torch.norm(u).item()
            self.norm.append([avg_norm, self.avg_malicious_norm, self.avg_benign_norm])

        for model_param, update_param in zip(self.global_model.parameters(), self.global_update):
            model_param.data += update_param.data.clone()

    def add_parameters(self, w, client_update):
        for server_param, client_param in zip(self.global_update, client_update):
            server_param.data += client_param.data.clone() * w