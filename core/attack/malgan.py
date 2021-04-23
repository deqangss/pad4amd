"""
malgan: https://arxiv.org/pdf/1702.05983.pdf
"""
import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import to_tensor, mkdir
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.malgan')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class MalGAN(BaseAttack, nn.Module):
    """
    malgan

    Parameters
    ---------
    @param input_dim, the size of an input
    @param noise_dim, the dimension of noise vector
    @param model_path, string, model path for saving the generator
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, input_dim, noise_dim=28, model_path=None,
                 is_attacker=True, kappa=1., manipulation_x=None, omega=None, device=None):
        BaseAttack.__init__(self, is_attacker, kappa, manipulation_x, omega, device)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.latent_dim = input_dim + noise_dim
        self.model_path = model_path

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 512),
            *block(512, 1024),
            nn.Linear(1024, self.input_dim),
            nn.Softmax()
        )
        self.generator.to(self.device)

    def forward(self, x):
        # x.shape [batch_size, N, vocab_dim]
        size = x.shape
        padding_mask = torch.sum(x, dim=-1, keepdim=True) > 1
        x_ext = torch.cat([x.reshape(size[0], -1), torch.rand((size[0], self.noise_dim), device=self.device)], dim=1)
        x_pertb = torch.round(self.generator(x_ext)).reshape(size) * padding_mask
        return torch.maximum(x, x_pertb)

    def fit(self, train_data_producer, validation_data_producer, detector, epochs=100, lr=0.001, lambda_=1e4, verbose=True):
        """
        train the generator for waging attack

        Parameters
        --------
        @param train_data_producer, Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param detector: a victim model
        @param epochs, Integer, epochs
        @param lr, Float, learning rate for Adam optimizer
        @param lambda_, Float, penalty factor
        @param verbose, Boolean, whether print details during the train phase
        """
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0
        if detector.k > 0:
            logger.warning("The attack leads to dense graph and trigger the issue of out of memory.")
        nbatches = len(train_data_producer)
        for i in range(epochs):
            self.train()
            losses, accuracies = [], []
            for idx_batch, (x_batch, adj, y_batch, _1) in enumerate(train_data_producer):
                x_batch, adj_batch, y_batch = to_tensor(x_batch, adj, y_batch, self.device)
                start_time = time.time()
                optimizer.zero_grad()
                x_pertb = self.forward(x_batch)
                loss_no_reduction, done = self.get_loss(detector, x_pertb, adj_batch, y_batch, lambda_)
                loss_train = torch.mean(loss_no_reduction)
                loss_train.backward()
                optimizer.step()
                total_time += time.time() - start_time
                acc_train = done.sum().item()
                acc_train /= x_batch.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, adj_val, y_val, _2 in validation_data_producer:
                    x_val, adj_val, y_val = to_tensor(x_val, adj_val, y_val, self.device)
                    x_val_pertb = self.forward(x_val)
                    _, done = self.get_loss(detector, x_val_pertb, adj_val, y_val, lambda_)
                    acc_val = done.sum().item()
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                if not path.exists(self.model_path):
                    mkdir(path.dirname(self.model_path))
                torch.save(self.generator.state_dict(), self.model_path)
                if verbose:
                    print(f'Model saved at path: {self.model_path}')

            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def perturb(self, x):
        self.generator.load_state_dict(torch.load(self.model_path))
        self.generator.eval()
        return self.forward(x)

    def get_loss(self, model, x_pertb, adj, label, lambda_=None):
        hidden, logit = model.forward(x_pertb, adj)
        ce = -1 * F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys():
            assert lambda_ is not None
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction = ce + lambda_ * (torch.clamp(
                    torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            else:
                loss_no_reduction = ce + self.lambda_ * (torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW))
            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done
