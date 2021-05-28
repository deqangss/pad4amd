from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time

from core.defense import Dataset
from core.defense import MalwareDetectorIndicator
from tools.utils import save_args, get_group_args, dump_pickle
from examples.maldet_test import cmd_md

indicator_argparse = cmd_md.add_argument_group(title='adv indicator')
indicator_argparse.add_argument('--beta', type=float, default=1., help='balance factor.')
indicator_argparse.add_argument('--sigma', type=float, default=0.1416,
                                help='standard deviation of isotropic Gaussian distribution.')
indicator_argparse.add_argument('--ratio', type=float, default=0.90,
                                help='ratio of validation examples remained for passing through malware detector')


def _main():
    args = cmd_md.parse_args()

    dataset = Dataset(k=args.k,
                      is_adj=args.is_adj,
                      n_sgs_max=args.N,
                      use_cache=args.cache,
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

    if args.mode == 'train':
        model.fit(train_dataset_producer,
                  val_dataset_producer,
                  epochs=args.epochs,
                  lr=args.lr,
                  weight_decay=args.weight_decay
                  )
        # human readable
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        # serialization for rebuilding neural nets
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

        # get threshold
        model.load()
        model.get_threshold(val_dataset_producer)
        model.save_to_disk()

    # human readable
    save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
    # serialization for rebuilding neural nets
    dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    # model.load()
    # print(model.tau)
    # # test: accuracy
    # model.predict(test_dataset_producer)


if __name__ == '__main__':
    _main()
