from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import numpy as np
import wandb
class Local(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientAVG)
        self.global_model = None
        print(f"\nTotal clients: {self.num_training_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds):

            for client in self.training_clients:
                client.train()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate local models for training clients ")
                self.evaluate()

        print(f"\n-------------Final Report-------------")
        print("\nFinal Personalized Accuracy: {:.2f}".format(self.rs_test_acc_p[-1]))
        

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        stats = np.array(stats).tolist()
        stats_train = np.array(stats_train).tolist()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        losses = [a / n for a, n in zip(stats_train[2], stats_train[1])]
        
        if acc == None:
            self.rs_test_acc_p.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss_p.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_test_accs_p.append(accs)

        print("Averaged Train Loss: {:.2f}".format(train_loss))
        print("Averaged Test Accurancy: {:.2f}%".format(test_acc*100))
        print("Std Test Accurancy: {:.2f}%".format(np.std(accs)*100))
        
        if self.detailed_info:
            print('Clients Train Loss:\n', [(idx, format(loss, '.2f')) for idx, loss in enumerate(losses)])
            print('Clients Test Accuracy:\n', [(idx, format(acc, '.2f')) for idx, acc in enumerate(accs)])

        try:
            if wandb.config['mode'] != 'debug':
                wandb.log({'p_train_loss':train_loss, 'p_test_acc':test_acc, 'p_std_test_acc':np.std(accs)})
        except Exception:
            pass