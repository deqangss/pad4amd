from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, KernelDensityEstimation, MalwareDetectorIndicator, MaxAdvTraining, PrincipledAdvTraining
from core.attack import MalGAN
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.malgan_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for l1 norm based projected gradient descent attack')
atta_argparse.add_argument('--lambda_', type=float, default=100000.,
                           help='balance factor for waging attack.')
atta_argparse.add_argument('--noise_dim', type=int, default=28,
                           help='dimension of noise vector.')
atta_argparse.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
atta_argparse.add_argument('--epochs', type=int, default=20,
                           help='number of epochs for training generator.')
atta_argparse.add_argument('--lr', type=float, default=0.001,
                           help='initial learning rate.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--kappa', type=float, default=10.,
                           help='attack confidence.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['maldet', 'kde', 'advmaldet', 'madvtrain', 'padvtrain'],
                           help="model type, either of 'maldet', 'advmaldet' and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'maldet':
        save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    elif args.model == 'kde':
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
    elif args.model == 'advmaldet':
        save_dir = config.get('experiments', 'malware_detector_indicator') + '_' + args.model_name
    elif args.model == 'madvtrain':
        save_dir = config.get('experiments', 'm_adv_training') + '_' + args.model_name
    elif args.model == 'padvtrain':
        save_dir = config.get('experiments', 'p_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'maldet', 'advmaldet' or 'padvtrain'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(k=hp_params['k'],
                      is_adj=hp_params['is_adj'],
                      n_sgs_max=hp_params['N'],
                      use_cache=hp_params['cache'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    (train_data, trainy), (val_data, valy), (test_data, testy) = \
        dataset.train_dataset, dataset.validation_dataset, dataset.test_dataset

    mal_train_data,  mal_trainy = train_data[trainy == 1], trainy[trainy == 1]
    mal_val_data, mal_valy = val_data[valy == 1], valy[valy == 1]
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_data, mal_testy = test_data[testy == 1], testy[testy == 1]
        from numpy import random
        mal_count = len(mal_testy) if len(mal_testy) < 1000 else 1000
        mal_test_data = random.choice(mal_test_data, mal_count, replace=False)
        mal_testy = mal_testy[:mal_count]
        utils.dump_pickle_frd_space((mal_test_data, mal_testy), mal_save_path)
    else:
        mal_test_data, mal_testy = utils.read_pickle_frd_space(mal_save_path)

    if len(mal_trainy) <= 0 or len(mal_valy) <= 0 or len(mal_testy) <= 0:
        return
    mal_train_dataset_producer = dataset.get_input_producer(mal_train_data, mal_trainy,
                                                            batch_size=hp_params['batch_size'],
                                                            name='train')
    mal_val_dataset_producer = dataset.get_input_producer(mal_val_data, mal_valy,
                                                          batch_size=hp_params['batch_size'],
                                                          name='val')
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_data, mal_testy,
                                                           batch_size=hp_params['batch_size'],
                                                           name='test')

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    if args.model == 'maldet' or args.model == 'kde':
        model = MalwareDetector(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    else:
        model = MalwareDetectorIndicator(vocab_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         sample_weights=dataset.sample_weights,
                                         name=args.model_name,
                                         **hp_params
                                         )
    model = model.to(dv)
    if args.model == 'kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()
    elif args.model == 'madvtrain':
        adv_model = MaxAdvTraining(model)
        adv_model.load()
        model = adv_model.model
    elif args.model == 'padvtrain':
        adv_model = PrincipledAdvTraining(model)
        adv_model.load()
        model = adv_model.model
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    model.predict(mal_test_dataset_producer)

    attack = MalGAN(input_dim=dataset.n_sgs_max * dataset.vocab_size,
                    noise_dim=args.noise_dim,
                    model_path=os.path.join(config.get('experiments', 'malgan'), 'gan-model.pkl'),
                    oblivion=args.oblivion,
                    kappa=args.kappa,
                    device=model.device
                    )
    model.eval()
    attack.fit(mal_train_dataset_producer,
               mal_val_dataset_producer,
               detector=model,
               epochs=args.epochs,
               lr=args.lr,
               lambda_=args.lambda_,
               verbose=True
               )
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []

    for i in range(hp_params['n_sample_times']):
        y_cent, x_density = [], []
        x_mod = []
        for x, a, y, g_ind in mal_test_dataset_producer:
            x, a, y = utils.to_tensor(x, a, y, model.device)
            adv_x_batch = attack.perturb(x)
            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch, a, y, use_indicator=True)
            y_cent.append(y_cent_batch)
            x_density.append(x_density_batch)
            x_mod.extend(dataset.get_modification(adv_x_batch, x, g_ind, True))
        y_cent_list.append(np.vstack(y_cent))
        x_density_list.append(np.concatenate(x_density))
        x_mod_integrated = dataset.modification_integ(x_mod_integrated, x_mod)
    y_cent = np.mean(np.stack(y_cent_list, axis=1), axis=1)
    y_pred = np.argmax(y_cent, axis=-1)
    logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / len(mal_testy) * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.mean(np.stack(x_density_list, axis=1), axis=1), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / len(mal_testy) * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / len(mal_testy) * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    save_dir = os.path.join(config.get('experiments', 'malgan'), args.model)
    if not os.path.exists(save_dir):
        utils.mkdir(save_dir)
    utils.dump_pickle_frd_space(x_mod_integrated,
                                os.path.join(save_dir, 'x_mod.list'))
    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               adj_mod=None)


if __name__ == '__main__':
    _main()
