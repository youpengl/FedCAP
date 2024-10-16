from flcore.clients.clientcap1 import clientCAP1
from flcore.servers.serverbase import Server
import os
import numpy as np
import torch
import copy
import math
import torch.nn.functional as F
from utils.byzantine import *
import time

class FedCAP1(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientCAP1)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

        self.phi = args.phi
        self.alpha = args.alpha
        self.alpha_name = self.alpha if args.alpha_name == '' else args.alpha_name
        self.normT = args.normT
        self.models_pool = []
        self.similarity_times = []
        self.aggregation_times = []

    def send_models(self):

        if self.uploaded_ids != []:
            '''Recover'''
            uploaded_models = []         
            for idx, update in enumerate(self.uploaded_updates):

                if self.models_pool != []:
                    model = copy.deepcopy(self.models_pool[idx])
                else:
                    model = copy.deepcopy(self.global_model)
                for model_param, update_param in zip(model.parameters(), update):
                    model_param.data += update_param.data.clone()

                uploaded_models.append(model)

            self.models_pool = []
            current_updates = self.uploaded_updates

            self.uploaded_weights = [weight/sum(self.uploaded_weights) for weight in self.uploaded_weights]
            global_model = copy.deepcopy(self.global_model)

            if self.current_round == self.global_rounds:
                self.save_global_model(current_updates, uploaded_models, global_model)


            all_invalid_true_ids = []
            similarity_times = []
            for c in self.training_clients:
                if c.itentity_as_malicious == True:
                    all_invalid_true_ids.append(c.id)
                else:
                    if c.id in self.uploaded_ids:
                        c_request_update = current_updates[self.uploaded_ids.index(c.id)]
                    else:
                        c_request_update = c.get_update(self.global_model)

                        if c.malicious == True and self.attack_type != 'A1':
                            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
                            c_request_update = eval(self.attack_type)(c_request_update, current_updates, malicious_ids, len(self.selected_clients))

                    coef = torch.zeros(len(current_updates))
                    index = None
                    u_i = torch.cat([p.data.reshape(1, -1) for p in c_request_update], dim=1)

                    '''Similarity Calculation'''
                    start = time.time()
                    for j, uu in enumerate(current_updates):
                        if c.id == self.uploaded_ids[j]:
                            coef[j] = -999.
                            index = j
                            continue
                        u_j = torch.cat([p.data.reshape(1, -1) for p in uu], dim=1)
                        simi = F.cosine_similarity(u_i, u_j)[0].item()
                        simi = -1 if math.isnan(simi) else simi
                        coef[j] = simi

                    coef = F.softmax(np.multiply(coef, self.alpha), dim=0).numpy()
                    if index != None:
                        if self.phi == -1:
                            self.phi = 1 / coef.shape[0]
                        coef = np.multiply(coef, 1-self.phi)
                        coef[index] = self.phi
                        
                    similarity_times.append(time.time()-start)

                    if c not in self.selected_clients:
                        c.resume_per_model = copy.deepcopy(c.c_per_model)

                    for param in c.c_per_model.parameters():
                        param.data.zero_()

                    for j, uu in enumerate(uploaded_models):
                        for param, param_j in zip(c.c_per_model.parameters(), uu.parameters()):
                            param.data += coef[j] * param_j.data.clone()

                    if c not in self.selected_clients:
                        c.resume_model = copy.deepcopy(c.model)
                        
                    c.set_parameters(c.c_per_model)

            self.similarity_times.append(sum(similarity_times))

            if self.attack_type != 'B':
                all_valid_true_ids = np.setdiff1d(self.training_clients_ids, all_invalid_true_ids).tolist()
                benign_ids = np.setdiff1d(self.training_clients_ids, self.malicious_ids).tolist()
                DAcc = (len([m_id for m_id in self.malicious_ids if m_id in all_invalid_true_ids])+ len([b_id for b_id in benign_ids if b_id in all_valid_true_ids]))/self.num_training_clients
                FPR = len([b_id for b_id in benign_ids if b_id in all_invalid_true_ids])/len(benign_ids)
                FNR = len([m_id for m_id in self.malicious_ids if m_id not in all_invalid_true_ids])/len(self.malicious_ids)
                print('Ours DAcc: {:.2f}%, FPR: {:.2f}%, FNR: {:.2f}%'.format(DAcc*100, FPR*100, FNR*100))
                
            for c in self.selected_clients:
                self.models_pool.append(c.c_per_model)
                
            start = time.time()
            self.aggregate_parameters(uploaded_models)
            self.aggregation_times.append(time.time()-start)
        
    def train(self):
        client_times = []
        for i in range(self.global_rounds+1):
            print(f"\n-------------Round number: {i}-------------")
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-Evaluate customized models for training clients")
                self.evaluate()

                if i > 0:
                    self.resume_client_model()
                    
                if i == self.global_rounds:
                    break
                
            start = time.time() 
            for client in self.selected_clients:
                client.train()

            client_times.append(time.time()-start)    

            self.receive_models()
            
        print(f"\n-------------Final Report-------------")
        print("\nFinal Testing Accuracy: {:.2f}%".format(self.rs_test_acc_g[-1]*100))

    def receive_models(self):

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_updates = []

        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_updates.append([c_param.data - s_param.data for c_param, s_param in zip(client.model.parameters(), client.c_per_model.parameters())])

        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.malicious_true_ids_current_round = [c_id for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(None, self.uploaded_updates, malicious_ids)
    
    def aggregate_parameters(self, uploaded_models):

        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def resume_client_model(self):
        for c in self.training_clients:
            if c not in self.selected_clients:
                try:
                    c.set_parameters(c.resume_model)
                    for old_param, new_param in zip(c.resume_per_model.parameters(), c.c_per_model.parameters()):
                        new_param.data = old_param.data.clone()
                except Exception:
                    pass 

    def save_global_model(self, current_updates, uploaded_models, global_model):
        model_path ="models/"

        filename = "{}_{}_{}_{}_{}_bz{}_lr{}_gr{}_ep{}_jr{}_nc{}_seed{}".format(self.dataset, self.partition, self.algorithm, 
                                                                                           self.attack_type, self.attack_ratio, self.batch_size, 
                                                                                           self.learning_rate, self.global_rounds, self.local_steps, 
                                                                                           self.join_ratio, self.num_clients, self.seed)

        filename = filename + "_lamda{}_alpha{}_phi{}_normT{}".format(self.args.lamda, self.alpha_name, self.phi, self.normT)

        filename = filename + '.tar'
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.model_path = os.path.join(model_path, filename)

        print("Model path: " + self.model_path)

        torch.save({'global_model': global_model,
                    'current_updates': current_updates,
                    'uploaded_models': uploaded_models}, self.model_path)
              