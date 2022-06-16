# -*- coding: utf-8 -*-
# !@time: 2022/6/16 下午1:23
# !@author: superMC @email: 18758266469@163.com
# !@fileName: save_2_dict.py


"""
存储ip_url
alink_url
"""
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
ip_url_list_dict = {}
for ip in ip_list:
    ip = filter_str(ip)
    for each_ip in ip:
        related_url = url_ip_url[url_ip_url['ip'].apply(lambda x: each_ip in x)]['url'].values.tolist()
        ip_url_list_dict[each_ip] = related_url
save_dict("data/ip_url_dict.json", ip_url_list_dict)

b_url_alink_url = b_url_alink_url.rename(columns={'b_url': 'url', 'b_a-domain': 'domain'})
p_url_alink_url = p_url_alink_url.rename(columns={'p_url': 'url', 'p_a-domain': 'domain'})

b_url_alink_url['domain'].map(
    lambda x: x.replace('\"', "").replace("\'", "").replace("[", "").replace("]", "").replace(" ", ""))

p_url_alink_url['domain'].map(
    lambda x: x.replace('\"', "").replace("\'", "").replace("[", "").replace("]", "").replace(" ", ""))

url_alink_url = pd.concat([b_url_alink_url, p_url_alink_url], axis=0).reindex()
p_url_alink_map = []
for alink in p_url_alink_url['domain']:
    if "" in p_url_alink_map:
        p_url_alink_map.remove("")
    if "." in alink:
        alink = filter_str(alink)
        p_url_alink_map += alink
p_url_alink_map = set(p_url_alink_map)
b_url_alink_map = []
for alink in b_url_alink_url['domain']:
    if "" in b_url_alink_map:
        b_url_alink_map.remove('')
    if "." in alink:
        alink = filter_str(alink)
        b_url_alink_map += alink
b_url_alink_map = set(b_url_alink_map)
alink_list = list(set.union(b_url_alink_map, p_url_alink_map))
alink_list.remove(".")
alink_url_list_dict = {}
i = 0
for alink in tqdm(alink_list):
    tmp = url_alink_url['domain'].apply(lambda x: alink in x)
    related_url_list = url_alink_url[tmp]['url'].values.tolist()
    alink_url_list_dict[alink] = related_url_list
    i += 1

save_dict("data/alink_url_dict.json", alink_url_list_dict)
