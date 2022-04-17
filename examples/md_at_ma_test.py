from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial

from core.defense import Dataset
from core.defense import DNNMalwareDetector, MaxAdvTraining
from core.attack import Max, PGD, PGDl1, StepwiseMax
from tools.utils import save_args, get_group_args, dump_pickle
from examples.amd_icnn_test import cmd_md

max_adv_argparse = cmd_md.add_argument_group(title='max adv training')
max_adv_argparse.add_argument('--beta', type=float, default=0.1, help='penalty factor on adversarial loss.')
max_adv_argparse.add_argument('--ma', type=str, default='max', choices=['max', 'stepwise_max'],
                              help="Type of mixture of attack: 'max' or 'stepwise_max' strategy.")
max_adv_argparse.add_argument('--m', type=int, default=20,
                              help='maximum number of perturbations.')
max_adv_argparse.add_argument('--steps_l2', type=int, default=50,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                              help='step length in each step.')
max_adv_argparse.add_argument('--steps_linf', type=int, default=100,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_linf', type=float, default=0.01,
                              help='step length in each step.')
max_adv_argparse.add_argument('--random_start', action='store_true', default=False,
                              help='randomly initialize the start points.')
max_adv_argparse.add_argument('--round_threshold', type=float, default=0.98,
                              help='threshold for rounding real scalars at the initialization step.')


def _main():
    args = cmd_md.parse_args()

    dataset = Dataset(use_cache=args.cache,
                      feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train')
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2 and args.epochs >= 5

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = DNNMalwareDetector(dataset.vocab_size,
                               dataset.n_classes,
                               device=dv,
                               name=model_name,
                               **vars(args)
                               )
    model = model.to(dv).double()

    pgdlinf = PGD(norm='linf', use_random=False,
                  is_attacker=False,
                  device=model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.steps_linf,
                              step_length=args.step_length_linf,
                              verbose=False
                              )
    pgdl2 = PGD(norm='l2', use_random=False, is_attacker=False, device=model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.steps_l2,
                            step_length=args.step_length_l2,
                            verbose=False
                            )
    pgdl1 = PGDl1(is_attacker=False, device=model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            m=args.m,
                            verbose=False)

    if args.ma == 'max':
        attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                     varepsilon=1e-9,
                     is_attacker=False,
                     device=model.device
                     )
        attack_param = {
            'steps_max': 1,  # steps for max attack
            'verbose': True
        }

    elif args.ma == 'stepwise_max':
        attack = StepwiseMax(is_attacker=False)
        attack_param = {
            'steps': max(max(args.m, args.steps_linf), args.steps_l2),
            'sl_l1': 1.,
            'sl_l2': args.step_length_l2,
            'sl_linf': args.step_length_linf
        }
    else:
        raise NotImplementedError("Expected 'max' and 'stepwise_max'.")

    max_adv_training_model = MaxAdvTraining(model, attack, attack_param)
    if args.mode == 'train':
        max_adv_training_model.fit(train_dataset_producer,
                                   val_dataset_producer,
                                   epochs=5,
                                   adv_epochs=args.epochs - 5,
                                   beta=args.beta,
                                   lr=args.lr,
                                   weight_decay=args.weight_decay
                                   )
        # human readable parameters
        save_args(path.join(path.dirname(max_adv_training_model.model_save_path), "hparam"), vars(args))
        # save parameters for rebuilding the neural nets
        dump_pickle(vars(args), path.join(path.dirname(max_adv_training_model.model_save_path), "hparam.pkl"))
    # test: accuracy
    max_adv_training_model.load()
    max_adv_training_model.model.predict(test_dataset_producer)

    # attr_cls, attr_de = max_adv_training_model.model.get_important_attributes(test_dataset_producer)
    # import numpy as np
    # np.save("./attributions-mad-cls", attr_cls)
    # np.save("./attributions-mad-de", attr_de)


if __name__ == '__main__':
    _main()
