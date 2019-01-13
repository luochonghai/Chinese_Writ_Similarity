'''win version'''
#coding:utf-8
import urllib.request
import re
import http.cookiejar
import execjs
import random
import time
import writ
import reading
import datetime
#import uuid
'''another way to get guid'''
#guid=uuid.uuid4()

def createGuid():
    return str(hex(int((1 + random.random()) * 0x10000) | 0))[3:]

def generate_guid():
    return createGuid() + createGuid() + "-" + createGuid() + "-" + createGuid() + createGuid() + "-" + createGuid() + createGuid() + createGuid(); 

def get_data():
    type_of_case = "刑事案件"
    case_type_list = {"刑事案件":'1',"民事案件":'2',"行政案件":'3',"赔偿案件":'4',"执行案件":'5'}
    trans_type_list = {"刑事案件":"%E5%88%91%E4%BA%8B%E6%A1%88%E4%BB%B6",
    "民事案件":"%E6%B0%91%E4%BA%8B%E6%A1%88%E4%BB%B6",
    "行政案件":"%E8%A1%8C%E6%94%BF%E6%A1%88%E4%BB%B6",
    "赔偿案件":"%E8%B5%94%E5%81%BF%E6%A1%88%E4%BB%B6",
    "执行案件":"%E6%89%A7%E8%A1%8C%E6%A1%88%E4%BB%B6"}
    eng_name_list = {"刑事案件":"crimi_case",
    "民事案件":"civil_case",
    "行政案件":"admin_case",
    "赔偿案件":"compe_case",
    "执行案件":"execu_case"}

    sta_date = '2018-07-01'
    end_date = '2018-07-03'
    date_sta = datetime.datetime.strptime(sta_date,'%Y-%m-%d')
    date_end = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    provinces = \
    ["最高人民法院","北京市","天津市","河北省","山西省",
    "内蒙古自治区","辽宁省","吉林省","黑龙江省","上海市",
    "江苏省","浙江省","安徽省","福建省","江西省",
    "山东省","河南省","湖北省","湖南省","广东省",
    "广西壮族自治区","海南省","重庆市","四川省","贵州省",
    "云南省","西藏自治区","陕西省","甘肃省","青海省",
    "宁夏回族自治区","新疆维吾尔自治区",
    "新疆维吾尔自治区高级人民法院生产建设兵团分院"
    ]
    #to construct the first param of Form_Data's parameter
    while date_sta < date_end:
        date_sta += datetime.timedelta(days = 1)
        data_head = date_sta.strftime('%Y-%m-%d')#from
        date_tail = (date_sta+datetime.timedelta(days = 1)).strftime('%Y-%m-%d')#to

        for prov in provinces:
            referer_str = "http://wenshu.court.gov.cn/List/List?sorttype=1&conditions=searchWord+++"+\
            data_head+"%20TO%20"+date_tail+"%E4%B8%8A%E4%BC%A0%E6%97%A5%E6%9C%9F:"+data_head+"%20TO%20"+date_tail+\
            "&conditions=searchWord+2+AJLX++"+"%E6%A1%88%E4%BB%B6%E7%B1%BB%E5%9E%8B:"+trans_type_list[type_of_case]

            guid = generate_guid()
            fh=open("D:\\FDU\\Template\\NLP(ZhipengXie)\\Spider\\"+eng_name_list[type_of_case]+"\\getkey.js","r")
            js_all = fh.read()
            fh.close()
            cjar=http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cjar))
            opener.addheaders=[("Referer",referer_str)]
            urllib.request.install_opener(opener)
            uapools=[
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393",
                "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.22 Safari/537.36 SE 2.X MetaSr 1.0",
                "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
                "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
                ]
            try:
                # fir_package = urllib.request.urlopen(referer_str)
                # res_num = fir_package.getcode()
                # if res_num != 200:
                #     print("Fir:%d"%(res_num))
                #     continue
                # fir_package.read().decode("utf-8","ignore")
                urllib.request.urlopen(referer_str).read().decode("utf-8","ignore")
                pat="vjkl5=(.*?)\s"
                vjkl5=re.compile(pat,re.S).findall(str(cjar))
                if(len(vjkl5)>0):
                    vjkl5=vjkl5[0]
                else:
                    vjkl5=0
                compile_js=execjs.compile(js_all)
                vl5x=compile_js.call("get_key",str(vjkl5))
                url="http://wenshu.court.gov.cn/List/ListContent"
                #100页,restricted by the website
                for i in range(0,2):
                    try:
                        codeurl="http://wenshu.court.gov.cn/ValiCode/GetCode"
                        codedata=urllib.parse.urlencode({
                                                "guid":guid,
                                                }).encode('utf-8')
                        codereq = urllib.request.Request(codeurl,codedata)
                        codereq.add_header('User-Agent',random.choice(uapools))
                        # sec_package = urllib.request.urlopen(codereq)
                        # resu_num = sec_package.getcode()
                        # if resu_num != 200:
                        #     print("Sec:%d"%(resu_num))
                        #     continue
                        # codedata = sec_package.read().decode("utf-8","ignore")
                        codedata=urllib.request.urlopen(codereq).read().decode("utf-8","ignore")
                        postdata =urllib.parse.urlencode({
                                                "Param":"上传日期:"+data_head+" TO "+date_tail+",案件类型:"+type_of_case+",法院地域:"+prov,
                                                "Index":str(i+1),
                                                "Page":"5",
                                                "Order":"法院层级",
                                                "Direction":"asc",
                                                "number":str(codedata),
                                                "guid":guid,
                                                "vl5x":vl5x,
                                                }).encode('utf-8')        
                        req = urllib.request.Request(url,postdata)
                        req.add_header('User-Agent',random.choice(uapools))
                        # thi_package = urllib.request.urlopen(req)
                        # resul_num = thi_package.getcode()
                        # if resul_num != 200:
                        #     print("Thi:%d"%(resul_num))
                        #     continue
                        # data = thi_package.read().decode("utf-8","ignore")
                        data=urllib.request.urlopen(req).read().decode("utf-8","ignore")
                        pat1='文书ID.*?".*?"(.*?)."'
                        pat2='裁判日期.*?".*?"(.*?)."'
                        pat3='案件名称.*?".*?"(.*?)."'
                        pat4='审判程序.*?".*?"(.*?)."'
                        pat5='案号.*?".*?"(.*?)."'
                        pat6='法院名称.*?".*?"(.*?)."'
                        allid1=re.compile(pat1).findall(data)
                        tian=open("D:\\FDU\\Template\\NLP(ZhipengXie)\\Spider\\"+eng_name_list[type_of_case]+"\\Doc_Id.txt",'a')
                        for j in allid1:
                            tian.write(j)
                            tian.write('\n')
                        tian.close()
                    except Exception as err:
                        print(err)

                    time.sleep(int((1+random.random())*5))
            except Exception as err:
                print(err)

if __name__ == '__main__':
    get_data()
    counter = writ.get_text()
    for i in range(counter):
        file_path = "ws_"+str(i)+".txt"
        reading.extract_text(file_path)