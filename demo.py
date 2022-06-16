# -*- coding: utf-8 -*-
# !@time: 2022/6/17 上午12:23
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import pandas as pd

b = pd.read_csv("data/benign/benign_merge_url.csv")
p = pd.read_csv("data/phishing/phishing_merge_url.csv")

b_url_set = pd.read_csv("data/benign/benign_merge_url.csv")['b_url'].values.tolist()
p_url_set = pd.read_csv("data/phishing/phishing_merge_url.csv")['p_url'].values.tolist()

filter = b_url_set + p_url_set
print(len(filter))
