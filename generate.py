# -*- coding: utf-8 -*-
# !@time: 2022/5/31 上午6:19
# !@author: superMC @email: 18758266469@163.com
# !@fileName: generate.py

from config import *
import pandas as pd


def main():
    data = pd.read_csv(domain, header=0)
    index = pd.read_csv(merge, header=0)[f'{prefix}_url']
    data[f'{prefix}_a-domain'] = data[f'{prefix}_a-domain'].map(
        lambda x: list(
            map(lambda y: y,
                x.replace("www.", "").replace("\'", "").replace("[", "").replace("]", "").replace(" ", "").split(','))))

    index_set = set(index.to_list())
    for i in range(len(data)):
        for element in data.iloc[i][f'{prefix}_a-domain']:
            index_set.add(element)
    index_list = sorted(list(index_set))
    pd.DataFrame(index_list).to_csv(id_csv, index=False, header=False)


id_csv = phishing_id
domain = phishing_domain
merge = phishing_merge
prefix = "p"
main()
