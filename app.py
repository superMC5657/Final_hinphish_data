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
from utils import setup
from utils.features_extraction_new import featureExtraction, parse_ip
from utils.parse_alink import parse_alink

whitelistDf = set(pd.read_csv(whitelist)["b_url"].tolist())
blacklistDf = set(pd.read_csv(blacklist)["p_url"].tolist())
app = Flask(__name__)
app.debug = True


def pre_test(url):
    if url in whitelistDf:
        return "white"
    elif url in blacklistDf:
        return "black"
    else:
        return "unknown"


def getDomain(url):
    pass


def getFeatures(url):
    pass


def getIp(url):
    pass


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
    print(url)
    res = pre_test(url)
    if res == "white":
        return "white"
    elif res == "black":
        return "black"
    else:
        ## 获取url的边信息
        domains = parse_alink(url)
        features = featureExtraction(url)
        ip = parse_ip(url)
        update_graph(url, related=domains)
        predict = inference(args, url, features, 2)
        if predict == 1:
            # white
            pass
        else:
            pass


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8012, threaded=False)
