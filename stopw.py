import jieba
TEXT_NUM = 5000
# jieba.load_userdict('userdict.txt')
# 创建停用词list
def stopwordslist(filepath):
    stopwords = open(filepath, 'r', encoding='gbk').read().split(" ")
    return stopwords
 
 
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopword.txt')  # 这里加载停用词的路径
    #result string
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t' and word != ' ' and word != '\n':
                outstr += word
                #outstr += " "
    return outstr
 
for i in range(TEXT_NUM): 
    inputs = open('datatemp/datatemp_'+str(i)+'_cooked.txt', 'r', encoding='gb18030')
    outputs = open('datastop/datastop_'+str(i)+'_cooked.txt', 'w')
    for line in inputs:
        line_seg = seg_sentence(line)  # 这里的返回值是字符串
        outputs.write(line_seg + '\n')
    outputs.close()
    inputs.close()
