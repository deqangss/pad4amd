from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time

from core.defense import Dataset
from core.defense import MalwareDetectorIndicator, PrincipledAdvTraining
from core.attack import OMPA
from tools.utils import save_args, get_group_args, dump_pickle
from examples.advdet_gmm_test import cmd_md

indicator_argparse = cmd_md.add_argument_group(title='towards principled adv training')
indicator_argparse.add_argument('--beta_a', type=float, default=0.005, help='penalty factor on adversarial loss.')
indicator_argparse.add_argument('--adv_epochs', type=int, default=20, help='epochs for adversarial training.')
indicator_argparse.add_argument('--m', type=int, default=10, help='maximum number of perturbations.')
indicator_argparse.add_argument('--step_length', type=float, default=1., help='step length.')
indicator_argparse.add_argument('--use_interdependent_api', action='store_true', default=False, help='whether consider the interdependent apis.')

def _main():
    args = cmd_md.parse_args()

    dataset = Dataset(args.dataset_name,
                      k=args.k,
                      is_adj=args.is_adj,
                      feature_ext_args=get_group_args(args, cmd_md, 'feature')
                      )
    (train_data, trainy), (val_data, valy), (test_data, testy) = dataset.train_dataset, dataset.validation_dataset, dataset.test_dataset
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
    attack = OMPA(is_attacker=False, use_dependent_api=args.use_interdependent_api, device=model.device)
    attack_param = {
        'm': args.m,
        'step_length': args.step_length,
        'verbose': False
    }
    principled_adv_training_model = PrincipledAdvTraining(model, attack, attack_param)

    if args.mode == 'train':
        principled_adv_training_model.fit(train_dataset_producer,
                                          val_dataset_producer,
                                          epochs=args.epochs,
                                          adv_epochs=args.adv_epochs,
                                          beta_a=args.beta_a,
                                          lr=args.lr,
                                          weight_decay=args.weight_decay
                                          )
        # human readable parameters
        save_args(path.join(path.dirname(principled_adv_training_model.model_save_path), "hparam"), vars(args))
        # save parameters for rebuilding the neural nets
        dump_pickle(vars(args), path.join(path.dirname(principled_adv_training_model.model_save_path), "hparam.pkl"))
    # test: accuracy
    principled_adv_training_model.model.load()
    principled_adv_training_model.model.predict(test_dataset_producer)


if __name__ == '__main__':
    _main()
