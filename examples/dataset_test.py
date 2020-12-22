from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.defense import Dataset


def main_():
    dataset = Dataset('drebin', is_adj=True)
    train_data, trainy = dataset.train_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=2, name='train')
    for _ in range(5):
        train_dataset_producer.reset_cursor()
        for idx, x, adj, l, sample_idx in train_dataset_producer.iteration():
            print(idx)
            print(len(x))
            print(x.shape)
            if dataset.is_adj:
                print(adj.shape)
            print(l)
            print(sample_idx.shape)


if __name__ == '__main__':
    main_()