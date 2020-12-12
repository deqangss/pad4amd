import argparse


cmd_md = argparse.ArgumentParser(description='Argparser for malware detector')
cmd_md.add_argument('--seed', type=int, default=0, help='random seed.')
cmd_md.add_argument('--embedding_dim', type=int, default=32, help='embedding dimension')
cmd_md.add_argument('--hidden_units', type=lambda s:[int(u) for u in s.split(',')], default='8', help='delimited list input, e.g., "32,32"',)
cmd_md.add_argument('--penultimate_hidden_dim', type=int, default=64, help='dimension of penultimate layer')
cmd_md.add_argument('--n_heads', type=int, default=2, help='number of headers')
cmd_md.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
cmd_md.add_argument('--k', type=int, default=10, help='sampling size')
cmd_md.add_argument('--alpha', type=float, default=0.2, help='slope coefficient of leaky-relu')
cmd_md.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')

cmd_md.add_argument('--batch_size', type=int, default=16, help='minibatch size')
cmd_md.add_argument('--epochs', type=int, default=100, help='number of epochs to train.')
cmd_md.add_argument('--lr', type=float, default=0.005, help='initial learning rate.')
cmd_md.add_argument('--patience', type=int, default=100, help='patience')
cmd_md.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')

# args = cmd_md.parse_args()


def build_kwargs(keys, arg_dict):
    st = ''
    for key in keys:
        st += '%s-%s' % (key, str(arg_dict[key]))
    return st


def save_args(fout, args):
    with open(fout, 'wb') as f:
        cp.dump(args, f, cp.HIGHEST_PROTOCOL)