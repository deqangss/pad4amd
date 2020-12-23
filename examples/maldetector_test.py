from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path

from core.defense import Dataset
from core.defense import MalwareDetector
from core.defense import defense_cmd_md
from core.droidfeature import feature_extraction_cmd_md

from tools.utils import save_args

all_args_dict = {}
defense_args = defense_cmd_md.parse_args()
defense_args_dict = vars(defense_args)
all_args_dict.update(defense_args_dict)

feat_args = feature_extraction_cmd_md.parse_args() # will be neglected
feat_args_dict = vars(feat_args)
all_args_dict.update(feat_args_dict)


def _main():
    dataset = Dataset('drebin', k=defense_args.k, use_cache=True, feature_ext_args=feat_args_dict)
    train_data, trainy = dataset.train_dataset
    val_data, valy = dataset.validation_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=4, name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=4, name='val')
    assert dataset.n_classes == 2

    dv = 'cpu'
    model = MalwareDetector(dataset.vocab_size, dataset.n_classes, device=dv, **defense_args_dict)
    model = model.to(dv)
    # dump hyper-parameters
    save_args(path.join(path.dirname(model.model_save_path), "hparam"), all_args_dict)
    model.fit(train_dataset_producer, val_dataset_producer, epochs=5)


if __name__ == '__main__':
    _main()
