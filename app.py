# -*- coding: utf-8 -*-
# !@time: 2022/6/8 下午11:25
# !@author: superMC @email: 18758266469@163.com
# !@fileName: app.py
import argparse

import flask
import pandas as pd
from flask import Flask, request
from infer import inference, update_graph
from config import whitelist, blacklist
from utils.util import setup, load_dict
from utils.features_extraction_new import featureExtraction
from utils.parse_ip import parse_ip
from utils.parse_alink import parse_alink

whitelistDf = set(pd.read_csv(whitelist)["b_url"].tolist())
blacklistDf = set(pd.read_csv(blacklist)["p_url"].tolist())
app = Flask(__name__)
app.debug = True
ip_url_list_dict: dict = load_dict("data/ip_url_dict.json")
alink_url_list_dict: dict = load_dict("data/alink_url_dict.json")


def pre_test(url, domains):
    if url in blacklistDf:
        return "phishing"

    for domain in domains:
        if domain in whitelistDf:
            return "safe"
    else:
        return "unknown"


def getArgs():
    parser = argparse.ArgumentParser('HAN')
    args = parser.parse_args().__dict__
    args['seed'] = 1
    args['log_dir'] = "results"
    args['dataset'] = "URL"
    args['hetero'] = True
    args = setup(args)
    return args


args = getArgs()


@app.route("/infer", methods=["GET", "POST"])
def post():
    print("处理中")
    url = request.form.get("url")
    domains = parse_alink(url)
    print("domains:", domains)
    print(url)
    res = pre_test(url, domains)
    if res == "safe":
        print("safe")
        return "safe"
    elif res == "phishing":
        print("phishing")
        return "phishing"
    else:
        ## 获取url的边信息
        ip = parse_ip(url)
        print(ip)
        # domain_url_list = parse_url_domain(domains)
        if ip == "":
            ip_url_list = []
        else:
            ip_url_list = ip_url_list_dict.get(ip)
        alink_url_list = []
        for alink in domains:
            url_list = alink_url_list_dict.get(alink)
            if url_list is not None:
                alink_url_list += url_list
        alink_url_list = list(set(alink_url_list))
        if ip_url_list is None:
            ip_url_list = []
            print("ip_url_list：None")
        else:
            print("ip_url_list: ", ip_url_list)

        print("alink_url_list：", alink_url_list)
        features = featureExtraction(url)
        update_graph(url, ip_url_list, type="ip")
        update_graph(url, alink_url_list, type="alink")
        predict = inference(args, url, features, 2)
        if predict == 1:
            return "safe"
        else:
            return "phishing"


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8012, threaded=False)
