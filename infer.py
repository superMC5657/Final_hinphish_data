import dgl
import torch
import pickle
from dgl.data.utils import download, get_download_dir, _get_dgl_url, save_graphs, load_graphs

from utils.parse_ip import parse_ip
from utils.util import load_label_url, get_binary_mask, EarlyStopping, load_dict
import numpy as np
import pandas as pd
from main import evaluate, score
from utils.features_extraction_new import featureExtraction
from utils.parse_alink import parse_alink

ip_urllist_dict = load_dict("data/ip_url_dict.json")


def update_graph(url: str, related: list, type='alink'):
    hg = load_graphs("data/backup/hg.bin")[0][0]
    with open('data/backup/id2url_map.pkl', 'rb') as f:
        id2url = pickle.load(f)

    with open('data/backup/url2id_map.pkl', 'rb') as f:
        url2id = pickle.load(f)

    old_node_list = list(id2url.keys())
    num_node = max(old_node_list)

    new_node_list = related.copy()
    new_node_list.append(url)
    for new_node in new_node_list:
        if new_node not in url2id.keys():
            new_node_id = num_node = num_node + 1
            id2url.update({new_node_id: new_node})
            url2id.update({new_node: new_node_id})
            hg = dgl.add_nodes(hg, 1, ntype='url')

    from_node = []
    to_node = []
    if type not in ['alink', 'ip']:
        print("error type")
    else:
        for to in related:
            from_node.append(url2id[url])
            to_node.append(url2id[to])
        dgl.add_edges(hg, from_node, to_node, etype=type)
    save_graphs("data/hg.bin", hg)
    with open('data/id2url_map.pkl', 'wb') as f:
        pickle.dump(id2url, f)
    with open('data/url2id_map.pkl', 'wb') as f:
        pickle.dump(url2id, f)


def inference(args, url, feat, num_classes=2):
    hg = load_graphs("data/hg.bin")[0][0]
    with open('data/id2url_map.pkl', 'rb') as f:
        id2url = pickle.load(f)

    with open('data/url2id_map.pkl', 'rb') as f:
        url2id = pickle.load(f)

    _, _, _, b_url_feature = load_label_url('benign')
    _, _, _, p_url_feature = load_label_url('phishing')
    b_url_feature = b_url_feature.rename(columns={'b_url': 'url'})
    p_url_feature = p_url_feature.rename(columns={'p_url': 'url'})
    url_feature = pd.concat([b_url_feature, p_url_feature], axis=0)
    url_feature['url_token'] = url_feature['url'].apply(lambda x: url2id[x])
    url_feature['feat'] = url_feature.iloc[:, 1:-2].values.tolist()
    url_feature = pd.DataFrame(url_feature, columns=['url_token', 'feat', 'label']).reindex()
    features = torch.FloatTensor(url_feature['feat'].values.tolist())
    url_feature[url_feature['label'] == -1] = 0
    labels = torch.LongTensor(url_feature['label'].values)

    float_mask = np.zeros(len(url_feature))
    for label in [1, -1]:
        mask = (url_feature['label'] == label).values
        float_mask[mask] = np.random.permutation(np.linspace(0, 1, mask.sum()))

    train_idx = np.where(float_mask <= 0.8)[0]
    val_idx = np.where((float_mask > 0.8) & (float_mask <= 0.9))[0]
    test_idx = np.where(float_mask > 0.9)[0]

    num_nodes = hg.number_of_nodes('url')
    train_idx = train_idx[train_idx < num_nodes]
    val_idx = val_idx[val_idx < num_nodes]
    test_idx = test_idx[test_idx < num_nodes]

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    from models.model_hetero import HAN
    model = HAN(meta_paths=[['alink'], ['ip']],
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    hg = hg.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss().to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    infer_feat = torch.FloatTensor(feat).to(args['device']).unsqueeze(0)
    all_feat = torch.cat([features, infer_feat], dim=0)
    labels = torch.cat([labels, torch.LongTensor([0])]).to(args['device'])
    train_mask = train_mask.add(False).to(args['device'])
    val_mask = val_mask.add(False).to(args['device'])
    test_mask = test_mask.add(False).to(args['device'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(hg, all_feat)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, hg, all_feat, labels, val_mask, loss_fcn)
        if epoch % 100 == 0:
            print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                  'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

    stopper.save_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, hg, all_feat, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

    infer_id = url2id[url]
    model.eval()
    logits = model(hg, all_feat)
    res = logits[infer_id]
    res = torch.argmax(res).detach().to('cpu').item()
    print(res)

    return res


def main(args):
    url = "https://www.baidu.com"
    domains = parse_alink(url)
    ip = parse_ip(url)

    domain_url_list = parse_url_domain(domains)
    ip_url_list = ip_urllist_dict[ip]
    features = featureExtraction(url)
    update_graph(url, [])

    res = inference(args, url, features)

    # res = inference(args, g, model, url, old_feat, feat)


if __name__ == '__main__':
    import argparse

    from utils.util import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('-d', '--dataset', type=str, default='ACM',
                        help='Dataset which model learned')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
