from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import argparse
import time

import torch
import torch.nn.functional as F

from core.defense import Dataset
from core.defense import MalwareDetectorIndicator
from tools.utils import save_args, get_group_args, to_tensor
from examples.maldet_test import cmd_md

#  alpha=1., sigma=0.7071,
indicator_argparse = cmd_md.add_argument_group(title='adv indicator')
indicator_argparse.add_argument('--beta', type=float, default=1., help='balance factor.')
indicator_argparse.add_argument('--sigma', type=float, default=0.15916,
                                help='standard deviation of isotropic Gaussian distribution, default value 1/sqrt(2)')


def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(args.dataset_name,
                      k=args.k,
                      use_cache=False,
                      is_adj=args.is_adj,
                      feature_ext_args=get_group_args(args, cmd_md, 'feature')
                      )
    train_data, trainy = dataset.train_dataset
    val_data, valy = dataset.validation_dataset
    test_data, testy = dataset.test_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=args.batch_size, name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(test_data, testy, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'
    #  beta, sigma, vocab_size, n_classes, n_sample_times=10, device='cpu', name='PRO',
    model = MalwareDetectorIndicator(vocab_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=time.strftime("%Y%m%d-%H%M%S"),
                                     **vars(args)
                                     )
    model = model.to(dv)
    save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
    model.fit(train_dataset_producer,
              val_dataset_producer,
              epochs=args.epochs,
              lr=args.lr,
              weight_decay=args.weight_decay
              )

    # test: accuracy
    model.predict(test_dataset_producer)
    # test: gradients of loss w.r.t. input
    model.adv_eval()
    for res in test_dataset_producer:
        x_batch, adj, y_batch, _1 = res
        x_batch, adj, y_batch = to_tensor(x_batch, adj, y_batch, dv)
        x_batch.requires_grad = True
        logits = model(x_batch, adj)[1]
        loss = F.cross_entropy(logits, y_batch)
        grad = torch.autograd.grad(loss, x_batch)[0]
        print(grad.shape)
        break


if __name__ == '__main__':
    _main()
