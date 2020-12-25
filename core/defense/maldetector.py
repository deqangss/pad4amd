import warnings
import time
from tqdm import tqdm
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from core.defense.malgat import MalGAT
from config import config
from tools import utils


class MalwareDetector(nn.Module):
    def __init__(self, vocab_size, n_classes, n_sample_times= 10, device='cpu', name='MALWARE_DETECTOR', **kwargs):
        """
        Construct malware detector
        :param vocab_size: Interger, the number of words in the vocabulary
        :param n_classes: Integer, the number of classes, n=2
        :param n_sample_times: Integer, the number of sampling times for predicting
        :param device: String, 'cpu' or 'cuda'
        :param name: String, model name
        """
        super(MalwareDetector, self).__init__()

        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.n_sample_times = n_sample_times
        self.device = device
        self.name = name
        self.parse_args(**kwargs)

        self.malgat = MalGAT(self.vocab_size,
                             self.embedding_dim,
                             self.n_hidden_units,
                             self.penultimate_hidden_unit,
                             self.n_heads,
                             self.dropout,
                             self.alpha,
                             self.k,
                             self.sparse)

        self.dense = nn.Linear(self.penultimate_hidden_unit, self.n_classes)
        self.model_save_path = path.join(config.get('experiments', self.name.lower()), 'model.pth')
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

    def parse_args(self,
                   embedding_dim=32,
                   n_hidden_units=None,
                   penultimate_hidden_unit=64,
                   n_heads=8,
                   dropout=0.6,
                   alpha=0.2,
                   k=10,
                   sparse=True,
                   **kwargs
                   ):
        self.embedding_dim = embedding_dim
        if n_hidden_units is None:
            self.n_hidden_units = [8]
        else:
            self.n_hidden_units = n_hidden_units
        self.penultimate_hidden_unit = penultimate_hidden_unit
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.k = k
        self.sparse = sparse
        if len(kwargs) > 0:
            warnings.warn("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, feature, adj=None):
        latent_representation = self.malgat(feature, adj)
        latent_representation = F.dropout(latent_representation, self.dropout, training=self.training)
        logits = self.dense(latent_representation)
        return latent_representation, logits

    def inference(self, test_data_producer):
        self.eval()
        test_data_producer.use_cache = True
        confidences = []
        gt_labels = []
        for _ in tqdm(range(self.n_sample_times)):
            conf_batchs = []
            for _1, x, adj, y, _2 in test_data_producer.iteration():
                x, adj, y = utils.to_tensor(x, adj, y, self.device)
                _, logits = self.forward(x, adj)
                conf_batchs.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)
            conf_batchs = torch.vstack(conf_batchs)
            confidences.append(conf_batchs)
            gt_labels = np.concatenate(gt_labels)
        confidences = torch.mean(torch.stack(confidences).permute([1, 0, 2]), dim=1)
        return confidences, gt_labels

    def predict(self, test_data_producer):
        # load model
        self.load_state_dict(torch.load(self.model_save_path))
        self.eval()
        # evaluation
        confidence, y_true = self.inference(test_data_producer)
        y_pred = (confidence.argmax(1) == y_true).item()
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)

        MSG = "The accuracy on the test dataset is {:.5f}%"
        print(MSG.format(accuracy * 100))
        MSG = "The balanced accuracy on the test dataset is {:.5f}%"
        print(MSG.format(b_accuracy * 100))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=5e-4, verbose=True):
        """
        Train the malware detector, pick the best model according to the cross-entropy loss on validation set
        :param train_data_producer: Object, an iterator for producing a batch of training data
        :param validation_data_producer: Object, an iterator for producing validation dataset
        :param verbose: Boolean, whether to show verbose logs
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_acc = 0.
        total_time = 0.
        for i in range(epochs):
            train_data_producer.reset_cursor()
            nbatchs = train_data_producer.mini_batches
            for idx_batch, x_batch, adj, y_batch, _1 in train_data_producer.iteration():
                print(y_batch)
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, self.device)
                self.train()
                start_time = time.time()
                optimizer.zero_grad()
                latent_rpst, logits = self.forward(x_batch, adj_batch)
                loss_train = F.cross_entropy(logits, y_batch)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time
                acc_train = (logits.argmax(1) == y_batch).sum().item()
                acc_train /= x_batch[0].size()[0]
                mins, secs = int(total_time) / 60, int(total_time) % 60
                print('Step: %d/%d' % (i * nbatchs + idx_batch + 1, epochs * nbatchs),
                      " | training time in %d minutes, %d seconds" % (mins, secs))
                print(f'\tTraining loss: {loss_train.item():.4f}\t|\t Train accuracy: {acc_train * 100:.2f}')

                self.eval()
                if (i * nbatchs + idx_batch + 1) % 100 == 0 and verbose:
                    avg_acc_val = []
                    validation_data_producer.reset_cursor()
                    with torch.no_grad():
                        for idx_batch, x_batch, adj, y_batch, _2 in validation_data_producer.iteration():
                            x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, self.device)
                            latent_rpst, logits = self.forward(x_batch, adj_batch)
                            acc_val = (logits.argmax(1) == y_batch).sum().item()
                            acc_val /= x_batch[0].size()[0]
                            avg_acc_val.append(acc_val)
                        avg_acc_val = np.mean(avg_acc_val)
                    if verbose:
                        print(f'\t Validation accuracy: {avg_acc_val * 100:.2f}')
                    if avg_acc_val >= best_acc:
                        torch.save(self.state_dict(), self.model_save_path)
                        if verbose:
                            print(f'\t Model saved at path: {self.model_save_path}')
