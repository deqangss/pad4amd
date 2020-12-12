from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.defense import Dataset
from core.defense import MalwareDetector
from core.defense import defense_cmd_md

defense_args = defense_cmd_md.parse_args()
defense_args_dict = vars(defense_args)


def main_():
    dataset = Dataset('drebin', k=defense_args.k)
    train_data, trainy = dataset.train_dataset
    val_data, valy = dataset.validation_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=4, name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=4, name = 'val')
    assert dataset.n_classes==2

    dv = 'cpu'
    model = MalwareDetector(dataset.vocab_size, dataset.n_classes, device=dv, **defense_args_dict)
    model = model.to(dv)
    model.fit(train_dataset_producer, val_dataset_producer, epochs=5)


if __name__ == '__main__':
    main_()