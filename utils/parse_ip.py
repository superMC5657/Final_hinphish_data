# 安装dnspython模块
import dns.resolver
import re

# 获取url地址响应的ip
def parse_ip(url):
    if not re.match(r"^https?", url):
        url = "http://" + url
    domain = re.findall(r"://([^/]+)/?", url)[0]
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")
    my_resolver = dns.resolver.Resolver(configure=False)
    my_resolver.nameservers = ['223.121.180.100', '8.8.8.8', '1.1.1.1']
    try:
        a = my_resolver.query('www.' + domain, 'A')
        for a_record in a.response.answer:
            for ip in a_record:
                ip = str(ip)
                # print(ip)
    except Exception as e:
        ip = ""
        print(str(e))
    return ip

if __name__ == '__main__':
    url = 'http://menon.org.gr/test'
    ip = parse_ip(url)
    print(ip)