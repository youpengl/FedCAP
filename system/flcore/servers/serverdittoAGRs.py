from flcore.clients.clientdittoAGRs import clientDittoAGRs
from flcore.servers.serverbase import Server
import copy
from utils.byzantine import *

class DittoAGRs(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientDittoAGRs)
        self.aggregation = args.aggregation
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\n-Evaluate received models for training clients")
                self.evaluate()

                self.resume_client_model()

            if i == self.global_rounds:
                break
                
            for client in self.selected_clients:
                client.dtrain()

            if i%self.eval_gap == 0:
                print("\n-Evaluate fine-tuned models for training clients ")
                self.evaluate_personalized()

            self.receive_models()
            self.aggregate_parameters()

        print(f"\n-------------Final Report-------------")
        print("\nFinal Testing Accuracy: {:.2f}%".format(max(self.rs_test_acc_g[-1]*100, self.rs_test_acc_p[-1]*100)))

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

        if self.aggregation == 'median':
            self.uploaded_updates = self.median(self.uploaded_updates, self.device)
            self.uploaded_weights = [1.]

        elif self.aggregation == 'trim':
            expected_num_malicious = int(len(self.selected_clients)*self.attack_ratio)
            m = int(expected_num_malicious/2)
            assert len(self.selected_clients) - 2 * m > 0
            self.uploaded_updates = self.trim(self.uploaded_updates, m, self.device)
            self.uploaded_weights = [1.]
        
        elif self.aggregation == 'cluster':
            self.uploaded_updates, self.uploaded_weights = self.cluster(self.uploaded_updates, self.uploaded_weights)

        else:
            self.uploaded_weights = [i/sum(self.uploaded_weights) for i in self.uploaded_weights]


        for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
            self.add_parameters(w, client_update)

        for model_param, update_param in zip(self.global_model.parameters(), self.global_update):
            model_param.data += update_param.data.clone()

    def add_parameters(self, w, client_update):
        for server_param, client_param in zip(self.global_update, client_update):
            server_param.data += client_param.data.clone() * w


    @staticmethod
    def cluster(uploaded_updates, uploaded_weights):
        num = len(uploaded_updates)
        flattened_grads = []
        for i in range(len(uploaded_updates)):
            flattened_grads.append(torch.cat([param.reshape(-1, 1) for param in uploaded_updates[i]], dim=0).squeeze())
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    flattened_grads[i], flattened_grads[j], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -np.inf] = -1
        dis_max[dis_max == np.inf] = 1
        dis_max[np.isnan(dis_max)] = -1

        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete", n_clusters=2
        )
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0

        uploaded_updates = [uploaded_updates[i] for i in range(num) if i in np.where(clustering.labels_==flag)[0]]
        uploaded_weights = [uploaded_weights[i] for i in range(num) if i in np.where(clustering.labels_==flag)[0]]

        uploaded_weights = [uploaded_weights[i]/sum(uploaded_weights) for i in range(len(uploaded_weights))]

        return uploaded_updates, uploaded_weights

    @staticmethod
    def median(uploaded_updates, device):

        num_layers = len(uploaded_updates[0])
        aggregated_param = []

        for i in range(num_layers):
            a = []
            for update in uploaded_updates:
                for j, param in enumerate(update):
                    if j == i:
                        a.append(param.clone().detach().cpu().numpy().flatten())
                        break
            aggregated_v = np.reshape(np.median(a, axis=0), newshape=param.shape)
            aggregated_param.append(torch.tensor(aggregated_v).to(device))

        return [aggregated_param]
    
    @staticmethod
    def trim(uploaded_updates, m, device):
        num_layers = len(uploaded_updates[0])
        aggregated_param = []

        for i in range(num_layers):
            a = []
            for update in uploaded_updates:
                for j, param in enumerate(update):
                    if j == i:
                        a.append(param.clone().detach().cpu().numpy().flatten())
                        break

            a = np.array(a)
            a = np.sort(a, axis=0)
            a = a[m:len(uploaded_updates)-m, :]
            a = np.mean(a, axis=0)     
            a = np.reshape(a, newshape=param.shape)
            aggregated_param.append(torch.tensor(a).to(device))

        return [aggregated_param]
