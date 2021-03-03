from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time

import torch

from core.defense import Dataset
from core.defense import MalwareDetectorIndicator, PrincipledAdvTraining
from core.attack import OMPA
from tools.utils import save_args, get_group_args, dump_pickle, read_pickle
from examples.advmaldet_test import cmd_md

indicator_argparse = cmd_md.add_argument_group(title='principled adv training')
indicator_argparse.add_argument('--lambda_', type=float, default=1., help='balance factor for waging attack.')
indicator_argparse.add_argument('--n_pertb', type=int, default=10, help='maximum number of perturbations.')


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

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = MalwareDetectorIndicator(vocab_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     sample_weights=dataset.sample_weights,
                                     name=model_name,
                                     **vars(args)
                                     )
    model = model.to(dv)
    attack = OMPA(lambda_=args.lambda_, n_perturbations=args.n_pertb, device=model.device)
    principled_adv_training_model = PrincipledAdvTraining(model, attack)

    if args.mode == 'train':
        principled_adv_training_model.fit(train_dataset_producer,
                                          val_dataset_producer,
                                          epochs=args.epochs,
                                          lr=args.lr,
                                          weight_decay=args.weight_decay
                                          )
        save_args(path.join(path.dirname(principled_adv_training_model.model_save_path), "hparam"), vars(args))

        # get threshold
        principled_adv_training_model.model.get_threshold(val_dataset_producer)
        print(principled_adv_training_model.model.tau)
        principled_adv_training_model.model.save_to_disk()
    # test: accuracy
    dump_pickle(vars(args), path.join(path.dirname(principled_adv_training_model.model_save_path), "hparam.pkl"))
    print(read_pickle(path.join(path.dirname(principled_adv_training_model.model_save_path), "hparam.pkl")))
    # principled_adv_training_model.model.predict(test_dataset_producer, use_indicator=True)


if __name__ == '__main__':
    _main()
