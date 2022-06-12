import dgl
import torch
import pickle
from dgl.data.utils import download, get_download_dir, _get_dgl_url, save_graphs, load_graphs


def update_graph(url: str, related: list, type='alink'):
    hg = load_graphs("data/hg.bin")[0][0]
    with open('data/id2url_map.pkl', 'rb') as f:
        id2url = pickle.load(f)

    with open('data/url2id_map.pkl', 'rb') as f:
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

def update_model():
    pass


def inference():
    pass


def main():
    url = "miracleyin.top"
    related = ["http://serv-acamai.com/1und1/de/index.php",
              "http://www.mariagraziagiove.com/ibdg/phpmailer/phpmailer/language/BLmsDqH6533S83cNf6c/"]
    update_graph(url, related)


if __name__ == '__main__':
    import argparse

    from utils import setup

    main()
