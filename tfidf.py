#coding=utf-8
import jieba
from gensim import corpora, models, similarities
TEXT_NUM = 500


def text_similarity():
    # gensim的模型model模块，可以对corpus进行进一步的处理，比如tf-idf模型，lsi模型，lda模型等
    wordstest_model = []
    for i in range(TEXT_NUM):
        file_line = open("datane/datane_"+str(i)+"_cooked.txt","r")
        wordstest_model.append(file_line.read())
    #wordstest_model = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    test_model = [[word for word in jieba.cut(words)] for words in wordstest_model]
    dictionary = corpora.Dictionary(test_model,prune_at=2000000)
    #for key in dictionary.iterkeys():
    #    print(key,dictionary.get(key),dictionary.dfs[key])
    corpus_model= [dictionary.doc2bow(test) for test in test_model]
    #print(corpus_model)

    # 目前只是生成了一个模型,并不是将对应的corpus转化后的结果,里面存储有各个单词的词频，文频等信息
    tfidf_model = models.TfidfModel(corpus_model)
    # 对语料生成tfidf
    corpus_tfidf = tfidf_model[corpus_model]

    text_simi = []
    #使用测试文本来测试模型，提取关键词,test_bow提供当前文本词频，tfidf_model提供idf计算
    for ind in range(TEXT_NUM):
        testword = wordstest_model[ind]
        test_bow = dictionary.doc2bow([word for word in jieba.cut(testword)])
        #print(test_bow)
        test_tfidf = tfidf_model[test_bow]
        most_fre = sorted(test_tfidf,key = lambda x:x[1],reverse = True)
        # 词id,tfidf值

        # 计算相似度
        index = similarities.MatrixSimilarity(corpus_tfidf) #把所有评论做成索引
        sims = index[test_tfidf]  #利用索引计算每一条评论和商品描述之间的相似度
        temp_max = 0
        temp_max_ind = 1
        text_simi.append(sims)
    return text_simi

            
