from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time

from core.defense import Dataset
from core.defense import AdvMalwareDetectorICNN, DNNMalwareDetector
from tools.utils import save_args, get_group_args, dump_pickle
from examples.nn_dnn_test import cmd_md

indicator_argparse = cmd_md.add_argument_group(title='adv indicator')
indicator_argparse.add_argument('--ratio', type=float, default=0.95,
                                help='ratio of validation examples remained for passing through malware detector')


def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(use_cache=args.cache, feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train')
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    md_model = DNNMalwareDetector(dataset.vocab_size,
                                  dataset.n_classes,
                                  device=dv,
                                  name=model_name,
                                  **vars(args)
                                  )
    model = AdvMalwareDetectorICNN(md_model,
                                   input_size=dataset.vocab_size,
                                   n_classes=dataset.n_classes,
                                   device=dv,
                                   name=model_name,
                                   **vars(args)
                                   )
    model = model.to(dv).double()

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

    model.load()
    # test: accuracy
    model.predict(test_dataset_producer)
    # attr_cls, attr_de = model.get_important_attributes(test_dataset_producer)
    # import numpy as np
    # np.save("./attributions-gmm-cls", attr_cls)
    # np.save("./attributions-gmm-de", attr_de)


if __name__ == '__main__':
    _main()
