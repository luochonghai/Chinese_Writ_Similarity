#coding:utf-8
import urllib.request
import re
import time
import requests
import http.cookiejar
#import execjs
import random
import sys,os
import reading

'''to get raw text from .aspx'''
def get_text():
    ws_url = "http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx"
    ws_title = "http://wenshu.court.gov.cn/content/content"
    fh = open("D:\\FDU\\Template\\NLP(ZhipengXie)\\Spider\\crimi_case\\Doc_Id.txt","r",encoding = 'gb18030',errors = 'ignore')
    doc_id = fh.readline()
    
    cjar=http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cjar))
    urllib.request.install_opener(opener)
     
    uapools=[
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.22 Safari/537.36 SE 2.X MetaSr 1.0",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
        ]
    #use counter to count how many wirts have been recorded
    counter = 0
    while doc_id:
        #use refresh_visit to revisit the website for info
        refresh_visit = 0
        try:
            target_url = ws_url+"?DocID="+doc_id
            target_title = ws_title+"?DocID="+doc_id
            codereq = urllib.request.Request(target_url)
            codereq.add_header('User-Agent',random.choice(uapools))
            data = urllib.request.urlopen(codereq).read().decode("utf-8","ignore")
            time.sleep(int((1+random.random())*5))
            # codetitle = urllib.request.Request(target_title)
            # codetitle.add_header('User-Agent',random.choice(uapools))
            # datatitle = urllib.request.urlopen(codetitle).read().decode("utf-8","ignore")
            if(len(data) > 50):
                ws_raw = open("D:\\FDU\\Template\\NLP(ZhipengXie)\\Spider\\crimi_case\\ws\\ws_"+str(counter)+".txt","a+",encoding = 'gb18030',errors = 'ignore')
                counter += 1
                print(counter)
                ws_raw.write(data)
                ws_raw.close()
                doc_id = fh.readline()
                refresh_visit = 0
            elif refresh_visit < 5:
                refresh_visit += 1
            else:
                refresh_visit = 0
                doc_id = fh.readline()
            # print(datatitle)
        except Exception as err:
            print(err)
    fh.close()
    return counter

# '''to extract useful info from .txt'''
# def extract_text(file_path):
#     fh = open(file_path,"r")
#     raw_text = fh.read()
#     if len(raw_text) < 50:
#         return 
#     fh.close()
#     temp_str_list = file_path.split(".")
#     new_file_path = temp_str_list[0]+"_cooked.txt"
#     fw = open(new_file_path,"a+")
#     str_match_list = \
#     [r"浏览：(.+?)次",r"结案方式\":(.+?),",r"补正文书\":(.+?),",
#     r"案件类型\":(.+?),",r"裁判日期\":(.+?),",r"文书ID\":(.+?)",
#     r"审判程序\":(.+?),",r"案号\":(.+?),",r"法院名称\":(.+?),",
#     r"法院区县\":(.+?),",r"法院ID\":(.+?),",r"DocContent\":(.+?),",
#     r"文书全文类型\":(.+?),",r"不公开理由\":(.+?),",r"法院地市\":(.+?),",
#     r"法院省份\":(.+?),",r"上传日期\":(.+?),",r"案件名称\":(.+?),",
#     r"法院区域\":(.+?),",r"文书类型\":(.+?),",r"效力层级\":(.+?),",
#     r"文本首部段落原文\":(.+?),",
#     r"诉讼参与人信息部分原文\":(.+?),",r"诉讼记录段原文\":(.+?)\",\"",
#     r"案件基本情况段原文\":(.+?),",r"裁判要旨段原文\":(.+?),",
#     r"判决结果段原文\":(.+?),",r"附加原文\":(.+?),",r"文本尾部原文\":(.+?)\}\)"
#     ]

#     str_title_list = \
#     ["浏览次数：","结案方式：","补正文书：",
#     "案件类型：","裁判日期：","文书ID：",
#     "审判程序：","案号：","法院名称：",
#     "法院区县：","法院ID：","DocContent：",
#     "文书全文类型：","不公开理由：","法院地市：",
#     "法院省份：","上传日期：","案件名称",
#     "法院区域：","文书类型：","效力层级：",
#     "文本首部段落原文：",
#     "诉讼参与人信息部分原文：","诉讼记录段原文：",
#     "案件基本情况段原文：","裁判要旨段原文：",
#     "判决结果段原文：","附加原文：","文本尾部原文："
#     ]

#     value_get_title = \
#     [r"hidDocID\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#hidCaseName",
#     r"hidCaseName\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#hidCaseNumber",
#     r"hidCaseNumber\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#hidCaseInfo",
#     r"hidCourt\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#hidCaseType",
#     r"hidCaseType\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#HidCourtID",
#     r"HidCourtID\"\)\.val\(\"(.+?)\"\)\;\$\(\"\#hidRequireLogin",
#     r"hidRequireLogin\"\)\.val\(\"(.+?)\"\)\;\}\)\;\$\(function\(\)\{var",
#     r"RelateInfo\:(.+?)\,LegalBase\:",
#     r"LegalBase\:(.+?)\}\]\}\]\}"
#     ]

#     value_get_res = \
#     ["hidDocID：","hidCaseName","hidCaseNumber",
#     "hidCourt：","hidCaseType：","HidCourtID：",
#     "hidRequireLogin：","RelateInfo：","LegalBase："
#     ]

#     str_type_cut = "</div><div style='LINE-HEIGHT: 25pt;TEXT-ALIGN:justify;TEXT-JUSTIFY:inter-"+\
#     "ideograph; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 仿宋; FONT-SIZE: 16pt;'>"
#     str_data_list = []
#     for i in range(len(str_title_list)):
#         fw.write(str_title_list[i]+re.findall(str_match_list[i],raw_text)[0]+'\n')

#     value_data_list = []
#     for j in range(len(value_get_title)):
#         if value_get_res[j] != "RelateInfo：" and value_get_res[j] != "LegalBase：":
#             fw.write(value_get_res[j]+re.findall(value_get_title[j],raw_text)[0]+'\n')
#         elif value_get_res[j] == "RelateInfo：":
#             name_list = []
#             value_list = []
#             cut_str = re.findall(value_get_title[j],raw_text)[0]
#             cut_list = cut_str.split("},{")
#             for cutted in cut_list:
#                 cutted_list = cutted.split(",")
#                 name_list.append(re.findall(r"(?<=\")[^}]*(?=\")",cutted_list[0])[0])
#                 cut_res = re.findall(r"\"(.+?)\"",cutted_list[-1])
#                 if len(cut_res) == 0:
#                     value_list.append("")
#                 else:
#                     value_list.append(cut_res[0])
#             for k in range(len(name_list)):
#                 fw.write(name_list[k]+":"+value_list[k]+'\n')
#         else:
#             cut_str = re.findall(value_get_title[j],raw_text)[0]
#             intab1 = "[ly]"
#             intab2 = "\u3000"
#             outtab1 = "    "
#             outtab2 = " "
#             trantab1 = str.maketrans(intab1,outtab1)
#             trantab2 = str.maketrans(intab2,outtab2)
#             cut_cooked = cut_str.translate(trantab1)
#             cut_cooked = cut_cooked.translate(trantab2)
#             regu_name_l = []
#             arti_name_l = []
#             arti_cont_l = []
#             #to get "法规名称" via searching substrings in the form of "《……》"
#             regu_name_p = re.findall(r"法规名称\:(.+?)Items{1}",cut_cooked)
#             for regu in regu_name_p:
#                 reg_c = re.findall(r"(?<=《)[^}]*(?=》)",regu)[0]
#                 regu_name_l.append(reg_c)
#             arti_name_p = re.findall(r"法条名称\:(.+?)法条内容{1}",cut_cooked)
#             for arti_n in arti_name_p:
#                 art_nc = re.findall(r"(?<=\')[^}]*(?=\')",arti_n)
#                 arti_name_l.append(art_nc[0])
#             arti_cont_p = re.findall(r"法条内容\:(.+?)\}\,\{{1}",cut_cooked)
#             for arti_space in arti_cont_p:
#                 arti_cont_l.append(arti_space.replace(' ',''))
#             fw.write("法规名称：\n")
#             for k in range(len(regu_name_l)):
#                 fw.write(regu_name_l[k]+'\n')
#             fw.write("法条名称：\n")
#             for k in range(len(arti_name_l)):
#                 fw.write(arti_name_l[k]+'\n')
#             fw.write("法条内容：\n")
#             for k in range(len(arti_cont_l)):
#                 fw.write(arti_cont_l[k]+'\n')

#     temp_start = []
#     temp_str = raw_text
#     test_str_len = len(str_type_cut)
#     fu = lambda x,y:0 if len(x) == 0 else x[-1]+y
#     while temp_str.find(str_type_cut) > -1:
#         right_value = temp_str.find(str_type_cut)
#         temp_start.append(right_value+fu(temp_start,test_str_len))
#         temp_str = temp_str[right_value+test_str_len:]
#     str_ridi = "</div><a type='dir' name='PJJG'></a><div style='LINE-HEIGHT: 25pt;TEXT-ALIGN:justify;"+\
#     "TEXT-JUSTIFY:inter-ideograph; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 仿宋; FONT-SIZE: 16pt;'>"
#     str_fin = "</div><a type='dir' name='WBWB'></a><br/><div style='TEXT-ALIGN: right; LINE-HEIGHT: 25pt; MAR"+\
#     "GIN: 0.5pt 72pt 0.5pt 0cm;FONT-FAMILY: 仿宋; FONT-SIZE: 16pt;'>"
#     str_d_t = "</div>\\\""
#     temp_last_b_w = raw_text.find(str_ridi)
#     temp_last_t = raw_text.find(str_fin)
#     temp_last_d = raw_text.find(str_d_t)
#     str_judge_text = raw_text[temp_start[-3]+test_str_len:temp_start[-2]]+\
#     raw_text[temp_start[-2]+test_str_len:temp_last_b_w]+\
#     raw_text[temp_last_b_w+len(str_ridi):temp_start[-1]]+\
#     raw_text[temp_start[-1]+test_str_len:temp_last_t]+\
#     raw_text[temp_last_t+len(str_fin):temp_last_d]
#     fw.write(str_judge_text+'\n')
#     fw.close()

if __name__ =="__main__":
    #counter = get_text()
    counter = 5000
    for i in range(counter):
        file_path = "ws/ws_"+str(i)+".txt"
        reading.extract_text(file_path)
