import torch
import os
import numpy as np
import copy
import wandb 
from utils.byzantine import *

class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.attack_ratio = args.attack_ratio
        self.attack_type = args.attack_type
        self.seed = args.seed
        self.algorithm = args.algorithm
        self.args = args
        self.current_round = -1
        self.num_training_clients = args.num_clients
        self.join_clients = int(self.num_training_clients * self.join_ratio)
        self.eval_gap = args.eval_gap
        self.detailed_info = args.detailed_info
        self.partition = args.partition
        self.data_path = args.data_path

        self.clients = []
        self.training_clients = []
        self.malicious_ids = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_updates = []
        
        self.rs_test_acc_g = []
        self.rs_train_loss_g = []
        self.rs_test_accs_g = []
        self.rs_test_acc_p = []
        self.rs_train_loss_p = []
        self.rs_test_accs_p = []

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
            self.clients.append(client)
        
        self.training_clients = self.clients
        self.training_clients_ids = np.arange(self.num_clients)

        print('Malicious Clients: {}'.format(list(self.malicious_ids)))


    def select_clients(self):
        
        selected_clients = list(np.random.choice(self.training_clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):

        for client in self.training_clients:

            if client not in self.selected_clients:
                client.resume_model = copy.deepcopy(client.model)

            client.set_parameters(self.global_model)


    def receive_models(self):

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []

        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
    
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):

        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path ="models/"
        
        filename = "{}_{}_{}_{}_{}_bz{}_lr{}_gr{}_ep{}_jr{}_nc{}_seed{}".format(self.dataset, self.partition, self.algorithm, 
                                                                                           self.attack_type, self.attack_ratio, self.batch_size, 
                                                                                           self.learning_rate, self.global_rounds, self.local_steps, 
                                                                                           self.join_ratio, self.num_clients, self.seed)

        if self.algorithm == 'FedAvg' and self.aggregation in ['mean', 'krum', 'mkrum', 'trim', 'cluster', 'median', 'rfa']:
            filename = filename + "_RAM{}".format(self.aggregation) # RAM: Robust Aggregation Method

        elif self.algorithm == 'Ditto':
            filename = filename + "_lamda{}".format(self.args.lamda)
        
        elif self.algorithm == 'DittoAGRs':
            filename = filename + "_lamda{}_RAM{}".format(self.args.lamda, self.aggregation)

        elif self.algorithm == 'FedFomo':
            filename = filename + "_O{}".format(self.O)
        
        filename = filename + '.pt'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.model_path = os.path.join(model_path, filename)
        print("Model path: " + self.model_path)
        torch.save({'global_model':self.global_model}, self.model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):

        filename = "{}_{}_{}_{}_{}_bz{}_lr{}_gr{}_ep{}_jr{}_nc{}_seed{}".format(self.dataset, self.partition, self.algorithm, 
                                                                                           self.attack_type, self.attack_ratio, self.batch_size, 
                                                                                           self.learning_rate, self.global_rounds, self.local_steps, 
                                                                                           self.join_ratio, self.num_clients, self.seed)

        if self.algorithm == 'FedAvg' and self.aggregation in ['mean', 'krum', 'mkrum', 'trim', 'cluster', 'median', 'rfa']:
            filename = filename + "_RAM{}".format(self.aggregation)
        
        elif self.algorithm == 'Ditto':
            filename = filename + "_lamda{}".format(self.args.lamda)
        
        elif self.algorithm == 'DittoAGRs':
            filename = filename + "_lamda{}_RAM{}".format(self.args.lamda, self.aggregation)

        elif self.algorithm == 'FedFomo':
            filename = filename + "_O{}".format(self.O)

        elif "FedCAP" in self.algorithm:
            filename = filename + "_lamda{}_alpha{}_phi{}_normT{}".format(self.args.lamda, self.alpha_name, self.phi, self.normT)
        
        result_path = "results/npz/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc_g) or len(self.rs_test_acc_p):
            file_path = result_path + "{}.npz".format(filename)
            print("Result path: " + file_path)
            np.savez(file_path, test_acc_g=self.rs_test_acc_g, 
                    test_acc_p=self.rs_test_acc_p, test_accs_g=self.rs_test_accs_g, 
                    test_accs_p=self.rs_test_accs_p, train_loss_g=self.rs_train_loss_g, 
                    train_loss_p=self.rs_train_loss_p)


    def test_metrics(self):
        num_samples = []
        tot_correct = []
        for c in self.training_clients:
            ct, ns = c.test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, tot_correct

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.training_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, losses
    
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        
        if self.malicious_ids != []:
            relative_malicious_ids = np.array([stats[0].index(i) for i in self.malicious_ids])

            stats_A = np.array(stats)[:, relative_malicious_ids].tolist()
            stats_train_A = np.array(stats_train)[:, relative_malicious_ids].tolist()

            test_acc_A = sum(stats_A[2])*1.0 / sum(stats_A[1])
            train_loss_A = sum(stats_train_A[2])*1.0 / sum(stats_train_A[1])

        else:
            test_acc_A = -1
            train_loss_A = -1

        benign_ids = np.sort(np.setdiff1d(self.training_clients_ids, self.malicious_ids))
        relative_benign_ids = np.array([stats[0].index(i) for i in benign_ids])

        stats_B = np.array(stats)[:, relative_benign_ids].tolist()
        stats_train_B = np.array(stats_train)[:, relative_benign_ids].tolist()

        stats = None
        stats_train = None

        test_acc = sum(stats_B[2])*1.0 / sum(stats_B[1])
        train_loss = sum(stats_train_B[2])*1.0 / sum(stats_train_B[1])
        accs = [a / n for a, n in zip(stats_B[2], stats_B[1])]

        
        if acc == None:
            self.rs_test_acc_g.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss_g.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_test_accs_g.append(accs)

        print("Benign Averaged Train Loss: {:.2f}".format(train_loss))
        print("Benign Averaged Test Accuracy: {:.2f}%".format(test_acc*100))
        print("Benign Std Test Accuracy: {:.2f}%".format(np.std(accs)*100))

        if test_acc >= 0.8: # Round-to-Accuracy
            print("Test accuracy >= 0.8!!!")

        if self.malicious_ids != []:
            print("Malicious Averaged Train Loss: {:.2f}".format(train_loss_A))
            print("Malicious Averaged Test Accuracy: {:.2f}%".format(test_acc_A*100))

        try:
            if wandb.config['mode'] != 'debug':
                if self.algorithm in ['FedAvg', 'Ditto', 'DittoAGRs', 'FedFomo', 'FedROD', 'FLTrust', 'FedCAP']:
                    wandb.log({'g_train_loss':train_loss, 'g_test_acc':test_acc, 'g_std_test_acc':np.std(accs)}, commit=False)
                else:
                    wandb.log({'g_train_loss':train_loss, 'g_test_acc':test_acc, 'g_std_test_acc':np.std(accs)})
        except Exception:
            pass

    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        if self.malicious_ids != []:
            relative_malicious_ids = np.array([stats[0].index(i) for i in self.malicious_ids])

            stats_A = np.array(stats)[:, relative_malicious_ids].tolist()
            stats_train_A = np.array(stats_train)[:, relative_malicious_ids].tolist()

            test_acc_A = sum(stats_A[2])*1.0 / sum(stats_A[1])
            train_loss_A = sum(stats_train_A[2])*1.0 / sum(stats_train_A[1])

        else:
            test_acc_A = -1
            train_loss_A = -1

        benign_ids = np.sort(np.setdiff1d(self.training_clients_ids, self.malicious_ids))
        relative_benign_ids = np.array([stats[0].index(i) for i in benign_ids])

        stats_B = np.array(stats)[:, relative_benign_ids].tolist()
        stats_train_B = np.array(stats_train)[:, relative_benign_ids].tolist()

        stats = None
        stats_train = None

        test_acc = sum(stats_B[2])*1.0 / sum(stats_B[1])
        train_loss = sum(stats_train_B[2])*1.0 / sum(stats_train_B[1])
        accs = [a / n for a, n in zip(stats_B[2], stats_B[1])]

        if acc == None:
            self.rs_test_acc_p.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss_p.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_test_accs_p.append(accs)

        print("Benign Averaged Train Loss: {:.2f}".format(train_loss))
        print("Benign Averaged Test Accurancy: {:.2f}%".format(test_acc*100))
        print("Benign Std Test Accurancy: {:.2f}%".format(np.std(accs)*100))

        if test_acc >= 0.8:
            print("Test accuracy >= 0.8!!!")
            
        if self.malicious_ids != []:
            print("Malicious Averaged Train Loss: {:.2f}".format(train_loss_A))
            print("Malicious Averaged Test Accurancy: {:.2f}%".format(test_acc_A*100))

        try:
            if wandb.config['mode'] != 'debug':
                wandb.log({'p_train_loss':train_loss, 'p_test_acc':test_acc, 'p_std_test_acc':np.std(accs)})
        except Exception:
            pass

    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        
        for c in self.training_clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)

            num_samples.append(ns)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        for c in self.training_clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, losses

    def resume_client_model(self):
        for c in self.training_clients:
            if c not in self.selected_clients:
                c.set_parameters(c.resume_model)
                c.resume_model = None