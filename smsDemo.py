"""
    @Project: PetraMind
    @File   : smsDemo.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-03-11
    @info   : 训练提示短信发送模块
    @Date   : 2025-07-03
    @info   : 修订为env文件保存短信发送用户参数。
"""

import os
from urllib import parse, request
from dotenv import load_dotenv
import hashlib   #加密
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--message", type=str, default='岩石图像工作站训练完成', help="")
opt = parser.parse_args()


statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词',
    '51': '手机号码不正确'
}


def md5s(strs):
   m = hashlib.md5()
   m.update(strs.encode("utf8")) #进行加密
   return m.hexdigest()


def sendMessage():

    load_dotenv("sms.env")

    smsapi = "http://api.smsbao.com/"

    # 短信平台账号
    user = os.getenv("SMS_USER", default="")
    sms_pass = os.getenv("SMS_PASS", default="")
    # 短信平台密码
    password = md5s(sms_pass)
    # 要发送的短信内容
    content = opt.message
    # 要发送短信的手机号码
    sms_phone = os.getenv("SMS_PHONE", default="")
    phone = str(sms_phone)

    data = parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content}) #参数
    send_url = smsapi + 'sms?' + data #拼接url
    print(send_url)
    response = request.urlopen(send_url) #发送请求
    the_page = response.read().decode('utf-8')
    # the_page = '40'
    try:
        print(statusStr[the_page])
    except:
        print('短信发送出现未知错误')


def main():
    sendMessage()


if __name__ == '__main__':
    main()