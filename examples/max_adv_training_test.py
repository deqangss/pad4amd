from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial

from core.defense import Dataset
from core.defense import MalwareDetector, MaxAdvTraining
from core.attack import Max, PGD, PGDl1, PGDAdam
from tools.utils import save_args, get_group_args, dump_pickle
from examples.maldet_test import cmd_md

max_adv_argparse = cmd_md.add_argument_group(title='max adv training')
max_adv_argparse.add_argument('--m', type=int, default=10, help='maximum number of perturbations.')

max_adv_argparse.add_argument('--n_step', type=int, default=50, help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_l2', type=float, default=2., help='step length in each step.')
max_adv_argparse.add_argument('--step_length_linf', type=float, default=0.02, help='step length in each step.')
max_adv_argparse.add_argument('--atta_lr', type=float, default=0.1, help='learning rate for pgd adam attack.')
max_adv_argparse.add_argument('--random_start', action='store_true', default=False, help='randomly initialize the start points.')
max_adv_argparse.add_argument('--round_threshold', type=float, default=0.98, help='threshold for rounding real scalars.')


def _main():
    args = cmd_md.parse_args()

    dataset = Dataset(args.dataset_name,
                      k=args.k,
                      is_adj=args.is_adj,
                      feature_ext_args=get_group_args(args, cmd_md, 'feature')
                      )
    (train_data, trainy), (val_data, valy), (
        test_data, testy) = dataset.train_dataset, dataset.validation_dataset, dataset.test_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=args.batch_size, name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(test_data, testy, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = MalwareDetector(dataset.vocab_size,
                            dataset.n_classes,
                            device=dv,
                            name=model_name,
                            **vars(args)
                            )
    model = model.to(dv)

    # initialize the base model of max attack
    pgdl1 = PGDl1(device=model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            m=args.m
                            )

    pgdl2 = PGD(norm='l2', use_random=False, device=model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.n_step,
                            step_length=args.step_length_l2,
                            verbose=False
                            )

    pgdlinf = PGD(norm='linf', use_random=args.random_start, rounding_threshold=args.round_threshold, device=model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.n_step,
                              step_length=args.step_length_linf,
                              verbose=False
                              )

    pgdadma = PGDAdam(use_random=args.random_start, rounding_threshold=args.round_threshold, device=model.device)
    pgdadma.perturb = partial(pgdadma.perturb,
                              steps=args.n_step,
                              lr=args.atta_lr,
                              verbose=False)

    attack = Max(attack_list=[pgdl1, pgdl2, pgdlinf, pgdadma],
                 varepsilon=1e-9,
                 device=model.device
                 )

    attack_param = {
        'steps': 1,
        'verbose': False
    }
    max_adv_training_model = MaxAdvTraining(model, attack, attack_param)

    if args.mode == 'train':
        max_adv_training_model.fit(train_dataset_producer,
                                   val_dataset_producer,
                                   epochs=args.epochs,
                                   lr=args.lr,
                                   weight_decay=args.weight_decay
                                   )
        # human readable parameters
        save_args(path.join(path.dirname(max_adv_training_model.model_save_path), "hparam"), vars(args))
        # save parameters for rebuilding the neural nets
        dump_pickle(vars(args), path.join(path.dirname(max_adv_training_model.model_save_path), "hparam.pkl"))
    # test: accuracy
    max_adv_training_model.model.load()
    max_adv_training_model.model.predict(test_dataset_producer)


if __name__ == '__main__':
    _main()
