import requests
import re
from lxml import etree

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"}

# 获取url相应的alink
def parse_alink(url):
    alink1 = []
    alink2 = []
    try:
        response = requests.get(url, headers=headers)
        content = response.content.decode()
        html = etree.HTML(content)
    except:
        html = ''
    a_link = html.xpath("//a/@href")
    # print(len(a_link))
    # 去重
    for a in a_link:
        if a not in alink1:
            alink1.append(a)
    # 保留alink的域名，同时去重
    for a in alink1:
        try:
            a = re.findall(r"://([^/]+)/?", a)[0]
        except:
            continue
        if a not in alink2:
            alink2.append(a)
    return alink2

if __name__ == '__main__':
    url = 'https://blog.csdn.net/zaf0516/article/details/123338865'
    alink = parse_alink(url)
    print(alink)