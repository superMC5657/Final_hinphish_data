# Purpose -
# Running this file (stand alone) - For extracting all the features from a web page for testing.
# Notes -
# 1 stands for legitimate
# 0 stands for suspicious
# -1 stands for phishing

import random

import html5lib.treewalkers.etree_lxml
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urlparse
import bs4
import pandas as pd
import re
import requests
import socket
import whois
from datetime import datetime
import time
import os
from lxml import html
from .patterns import *

import csv
import dns.resolver


headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"}

# 1.域名部分是否是IP地址
def having_ip_address(url):
    ip_address_pattern = ipv4_pattern + "|" + ipv6_pattern
    match = re.search(ip_address_pattern, url)
    return -1 if match else 1

# 2.域名长度
def url_length(url):
    length = len(url)
    return length

# 3.URL的深度
def getDepth(url):
  s = urlparse(url).path.split('/')
  depth = 0
  for j in range(len(s)):
    if len(s[j]) != 0:
      depth = depth+1
  return depth

# 4.是否使用URL短地址服务“Tiny URL”
def shortening_service(url):
    match = re.search(shortening_services, url)
    return -1 if match else 1

# 5.URL中是否有“@”
def having_at_symbol(url):
    match = re.search('@', url)
    return -1 if match else 1

# 6.重定向URL中的“//”位置
def double_slash_redirecting(url):
    pos = url.rfind('//')
    if pos > 6:
        if pos > 7:
            return -1
        else:
            return 1
    else:
        return 1

# 7.域名部分是否有“-”
def prefix_suffix(domain):
    match = re.search('-', domain)
    return -1 if match else 1

# 8.URL中“.”的个数
def having_sub_domain(url):
    if having_ip_address(url) == -1:
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5]))|(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',
            url)
        pos = match.end()
        url = url[pos:]
    num_dots = [x.start() for x in re.finditer(r'\.', url)]
    if len(num_dots) <= 3:
        return 1
    elif len(num_dots) == 4:
        return 0
    else:
        return -1

# 9.域名中是否含有“http/https”
def https_token(url):
    match = re.search(http_https, url)
    if match and match.start() == 0:
        url = url[match.end():]
    match = re.search('http|https', url)
    return -1 if match else 1

# 10.Favicon图标地址是否属于本域名
def favicon(wiki, soup, domain):
    for head in soup.find_all('head'):
        for head.link in soup.find_all('link', href=True):
            dots = [x.start() for x in re.finditer(r'\.', head.link['href'])]
            return 1 if wiki in head.link['href'] or len(dots) == 1 or domain in head.link['href'] else -1
    return 1

# 11.<img>、<audio>、<embed>、<i_frame>标签的外链比例
def request_url(wiki, soup, domain):
    i = 0
    success = 0
    for img in soup.find_all('img', src=True):
        dots = [x.start() for x in re.finditer(r'\.', img['src'])]
        if wiki in img['src'] or domain in img['src'] or len(dots) == 1:
            success = success + 1
        i = i + 1

    for audio in soup.find_all('audio', src=True):
        dots = [x.start() for x in re.finditer(r'\.', audio['src'])]
        if wiki in audio['src'] or domain in audio['src'] or len(dots) == 1:
            success = success + 1
        i = i + 1

    for embed in soup.find_all('embed', src=True):
        dots = [x.start() for x in re.finditer(r'\.', embed['src'])]
        if wiki in embed['src'] or domain in embed['src'] or len(dots) == 1:
            success = success + 1
        i = i + 1

    for i_frame in soup.find_all('i_frame', src=True):
        dots = [x.start() for x in re.finditer(r'\.', i_frame['src'])]
        if wiki in i_frame['src'] or domain in i_frame['src'] or len(dots) == 1:
            success = success + 1
        i = i + 1

    try:
        percentage = success / float(i) * 100
    except:
        return 1

    if percentage < 22.0:
        return 1
    elif 22.0 <= percentage < 61.0:
        return 0
    else:
        return -1

# 12.<a>标签的外链比例
def url_of_anchor(wiki, soup, domain):
    i = 0
    unsafe = 0
    for a in soup.find_all('a', href=True):
        if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (
                wiki in a['href'] or domain in a['href']):
            unsafe = unsafe + 1
        i = i + 1
        # print a['href']
    try:
        percentage = unsafe / float(i) * 100
    except:
        return 1
    if percentage < 31.0:
        return 1
        # return percentage
    elif 31.0 <= percentage < 67.0:
        return 0
    else:
        return -1

# 13.<Meta>, <Script> 和 <Link>标签外链比例
def links_in_tags(wiki, soup, domain):
    i = 0
    success = 0
    for link in soup.find_all('link', href=True):
        dots = [x.start() for x in re.finditer(r'\.', link['href'])]
        if wiki in link['href'] or domain in link['href'] or len(dots) == 1:
            success = success + 1
        i = i + 1

    for script in soup.find_all('script', src=True):
        dots = [x.start() for x in re.finditer(r'\.', script['src'])]
        if wiki in script['src'] or domain in script['src'] or len(dots) == 1:
            success = success + 1
        i = i + 1
    try:
        percentage = success / float(i) * 100
    except:
        return 1

    if percentage < 17.0:
        return 1
    elif 17.0 <= percentage < 81.0:
        return 0
    else:
        return -1

# 14.表单服务器处理（Server Form Handler ）
def sfh(wiki, soup, domain):
    if len(soup.find_all('form', action=True)) == 0:
        return 1
    else:
        for form in soup.find_all('form', action=True):
            if form['action'] == "" or form['action'] == "about:blank":
                return -1
            elif wiki not in form['action'] and domain not in form['action']:
                return 0
            else:
                return 1

# 15.网站重定向（Website Forwarding）
def forwarding(url):
    try:
        response = requests.get(url, headers=headers)
    except:
        response = " "
    if response == " ":
        return -1
    else:
        if len(response.history) <= 1:
            return 1
        elif 2<= len(response.history) <= 4:
            return 0
        else:
            return -1

# 16.状态栏变化
def mouseOver(content):
        if re.findall("<script>.+onmouseover.+</script>", content):
            return -1
        else:
            return 1


# 17.禁用鼠标右键
def rightClick(content):
    if re.findall(r"event.button ?== ?2", content):
      return 1
    else:
      return -1


# 18.使用弹窗
def popwindow(content):
        if re.findall(r"alert\(", content):
            return 1
        else:
            return -1

# 19. IFrame Redirection
def i_frame(soup):
    for i_frame in soup.find_all('i_frame', width=True, height=True, frameBorder=True):
        # Even if one iFrame satisfies the below conditions, it is safe to return -1 for this method.
        if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['frameBorder'] == "0":
            return -1
        if i_frame['width'] == "0" or i_frame['height'] == "0" or i_frame['frameBorder'] == "0":
            return 0
    # If none of the iframes have a width or height of zero or a frameBorder of size 0, then it is safe to return 1.
    return 1

# 20.dom长度
def dom_length(content):
    root = html.fromstring(content)
    tree = root.getroottree()
    result = root.xpath('//*')
    all_xpath_list = []
    for r in result:
        all_xpath_list.append(tree.getpath(r))
    tree_deeph_list = []
    for one_xpath in all_xpath_list:
        tree_deeph_list.append(len(one_xpath.split('/')[1:]))
    lennum = len(tree_deeph_list)
    return lennum

# 21.dom深度
def dom_deepth(content):
    root = html.fromstring(content)
    tree = root.getroottree()
    result = root.xpath('//*')
    all_xpath_list = []
    for r in result:
        all_xpath_list.append(tree.getpath(r))
    tree_deeph_list = []
    for one_xpath in all_xpath_list:
        tree_deeph_list.append(len(one_xpath.split('/')[1:]))
    maxnum = max(tree_deeph_list)
    return maxnum

# 22.域名时间
def domainAge(domain_name):
  try:
    whois_response = whois.whois(domain_name)
  except:
    return -1
  creation_date = whois_response.creation_date
  expiration_date = whois_response.expiration_date
  if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
    try:
      creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return -1
  if ((expiration_date is None) or (creation_date is None)):
      return -1
  elif ((type(expiration_date) is list) or (type(creation_date) is list)):
      return -1
  else:
    ageofdomain = abs((expiration_date - creation_date).days)
    if ((ageofdomain/30) < 6):
      age = -1
    else:
      age = 1
    return age


# 23.DNS记录
def DNS_record(domain):
    try:
        whois_response = whois.whois(domain)
        dns = 1
    except:
        dns = -1
    return dns

# 24.流量排名
def web_traffic(url):
    try:
        delay_time = random.randint(1,2)
        time.sleep(delay_time)
        print(f'等待{delay_time}s的时间爬流量排名...')
        rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
            "REACH")['RANK']
        rank = int(rank)
    except TypeError as t:
        print(t)
        return -1
    return 1 if rank < 100000 else 0
    # return -1

# 25.Page_Rank
def page_rank(domain):
    rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
        "name": domain
    })
    try:
        global_rank = int(re.findall(
            r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        global_rank = -1
    try:
        if global_rank > 0 and global_rank < 100000:
            global_rank = -1
        else:
            global_rank = 1
    except:
        global_rank = 1
    return global_rank


# 26.google_index
def google_index(url):
    site = (url, 5)
    return 1 if site else -1

# 27.Links_pointing_to_page
def Links_pointing_to_page(content):
    number_of_links = len(re.findall(r"<a href=", content))
    if number_of_links == 0:
        return 1
    elif number_of_links <= 2:
        return 0
    else:
        return -1

# 28. Statistical_report
def statistical_report(url,ip):
    url_match = re.search(
        'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly', url)
    try:
        ip_match = re.search(
            '146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
            '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
            '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
            '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
            '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
            '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42',
            ip)
        if url_match:
            return -1
        elif ip_match:
            return -1
        else:
            return 1
    except:
        print('Connection problem. Please check your internet connection')


# 获取domain
def parse_domain(url):
    if not re.match(r"^https?", url):
        url = "http://" + url
    domain = re.findall(r"://([^/]+)/?", url)[0]
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")
    return domain

# 获取html源码
def parse_html(url):
    try:
        response = requests.get(url, headers=headers)
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
    except:
        response = ""
        soup = -999
    return response, content, soup


# 获取url地址响应的ip
def parse_ip(domain):
    my_resolver = dns.resolver.Resolver(configure=False)
    my_resolver.nameservers = ['223.121.180.100', '8.8.8.8', '1.1.1.1']
    try:
        a = my_resolver.query('www.' + domain, 'A')
        for a_record in a.response.answer:
            for ip in a_record:
                ip = str(ip)
    except Exception as e:
        print(str(e))
        ip = ""
    return ip

# 特征提取
def featureExtraction(url):
    features = []
    # 获取domain
    domain = parse_domain(url)
    # 获取html源码
    result = parse_html(url)
    response = result[0]
    content = result[1]
    soup = result[2]
    # 获取ip
    ip = parse_ip(domain)

    # url特征
    features.append(having_ip_address(url))
    features.append(url_length(url))
    features.append(getDepth(url))
    features.append(shortening_service(url))
    features.append(having_at_symbol(url))
    features.append(double_slash_redirecting(url))
    features.append(prefix_suffix(domain))
    features.append(having_sub_domain(url))
    features.append(https_token(url))

    # html内容特征
    features.append(-1 if soup == -999 else favicon(url, soup, domain))
    features.append(-1 if soup == -999 else request_url(url, soup, domain))
    features.append(-1 if soup == -999 else url_of_anchor(url, soup, domain))
    features.append(-1 if soup == -999 else links_in_tags(url, soup, domain))
    features.append(-1 if soup == -999 else sfh(url, soup, domain))
    features.append(forwarding(url))
    features.append(-1 if response == "" else mouseOver(content))
    features.append(-1 if response == "" else rightClick(content))
    features.append(-1 if response == "" else popwindow(content))
    features.append(-1 if soup == -999 else i_frame(soup))
    features.append(-1 if response == "" else dom_length(content))
    features.append(-1 if response == "" else dom_deepth(content))

    # 第三方信息特征
    features.append(domainAge(domain))
    features.append(DNS_record(domain))
    features.append(web_traffic(url))
    features.append(page_rank(domain))
    features.append(google_index(url))
    features.append(Links_pointing_to_page(content))
    features.append(-1 if ip == "" else statistical_report(url,ip))

    print(features)
    # print(len(features))
    return features

if __name__ == '__main__':
    url = 'https://www.baidu.com'
    featureExtraction(url)
