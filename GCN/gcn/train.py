import dgl
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcn import GCN
# from gcn_mp import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    pass


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        pass

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    features = g.ndata['feat']
    labels = g.ndata['label']

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # create GCN model
    model = GCN(g, features.shape[1], args.n_hidden, data.num_labels, args.n_layers, F.relu, args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | ". format(epoch, loss.item(), acc))
        pass

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
    pass

