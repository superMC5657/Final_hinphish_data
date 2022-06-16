# -*- coding: utf-8 -*-
# !@time: 2022/6/16 下午1:23
# !@author: superMC @email: 18758266469@163.com
# !@fileName: save_2_dict.py
import pandas as pd
from tqdm import tqdm

import json
import datetime
import numpy as np


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(JsonEncoder, self).default(obj)


def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename, 'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)


from utils.util import load_label_url, filter_str

b_url_ip, b_url_alink_url, b_url_ip_url, b_url_feature = load_label_url('benign')
p_url_ip, p_url_alink_url, p_url_ip_url, p_url_feature = load_label_url('phishing')
b_url_ip_url = b_url_ip_url.rename(columns={'b_url': 'url', 'b_ip': 'ip'})
p_url_ip_url = p_url_ip_url.rename(columns={'p_url': 'url', 'p_ip': 'ip'})
url_ip_url = pd.concat([b_url_ip_url, p_url_ip_url], axis=0).reindex()
p_url_ip_map = []
for ip in p_url_ip_url['ip'].values.tolist():
    p_url_ip_map += ip
p_url_ip_map = set(p_url_ip_map)
b_url_ip_map = set(ip for ip in b_url_ip_url['ip'].values.tolist())
ip_list = list(set.union(p_url_ip_map, b_url_ip_map))
ip_urllist_dict = {}
for ip in tqdm(ip_list):
    ip = filter_str(ip)
    for each_ip in ip:
        related_url = url_ip_url[url_ip_url['ip'].apply(lambda x: each_ip in x)]['url'].values.tolist()
        ip_urllist_dict[each_ip] = related_url
save_dict("../data/ip_url_dict.json", ip_urllist_dict)
