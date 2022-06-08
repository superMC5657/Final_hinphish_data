import re
import requests
from bs4 import BeautifulSoup
# 安装dnspython模块
import dns.resolver

url = 'https://www.baidu.com'

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"}


# 匹配url中的域名（domain用于特征提取）
def parse_domain(url):
    if not re.match(r"^https?", url):
        url = "http://" + url
    domain = re.findall(r"://([^/]+)/?", url)[0]
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")
    return domain


# 获取url地址相应的HTML源码(soup用于特征提取)
def parse_html(url):
    try:
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        response = ""
        soup = -999
    return soup


# 获取url地址响应的ip
def parse_ip(domain):
    my_resolver = dns.resolver.Resolver(configure=False)
    my_resolver.nameservers = ['223.121.180.100', '8.8.8.8', '1.1.1.1']
    try:
        a = my_resolver.query('www.' + domain, 'A')
        for a_record in a.response.answer:
            for ip in a_record:
                ip = str(ip)
                print(ip)
    except Exception as e:
        print(str(e))



# 获取url相应的alink
def parse_alink(url):
    alink1 = []
    alink2 = []
    try:
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        soup = ''
    a_link = soup.xpath("//a/@href")
    alink1.append(a_link)
    for a in alink1:
        a = re.findall(r"://([^/]+)/?", a)[0]
        alink2.append(a)
    return alink2

