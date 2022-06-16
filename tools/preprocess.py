import torch
import pandas as pd
from config import *
from collections import defaultdict


def main():
    data = pd.read_csv(domain, header=0)
    index = pd.read_csv(merge, header=0)[f'{prefix}_url']
    index_list = sorted(list(set(index.to_list())))
    url_index_dict = defaultdict(int)
    for i in range(len(index_list)):
        url_index_dict[index_list[i]] = i + 1
        if i + 1 == 23829:
            print(index_list[i])
    data[f'{prefix}_url'] = data[f'{prefix}_url'].map(url_index_dict)
    data[f'{prefix}_a-domain'] = data[f'{prefix}_a-domain'].map(
        lambda x: list(
            map(lambda y: y,
                x.replace('\"', "").replace("\'", "").replace("[", "").replace("]", "").replace(" ", "").split(','))))

    print(data)
    print()


domain = phishing_domain
merge = phishing_merge
prefix = "p"
main()
