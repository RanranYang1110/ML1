#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_pc1006_nlp.py
# @time: 2019/04/03
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba
import os
import csv
import re
import sys
sys.path.append(r"")


class Similarity(object):
    """
        Attributes:

    """

    def __init__(self):
        self.path = r"D:\2-烟草行业\4-专家系统\ZB45包装机专家系统项目材料\开发文档\算法设计\DiagnoseServer\test\lib\data"
        # self.path = os.path.split(os.path.realpath(__file__))[0]
        self.jieba = jieba
        self.jieba.load_userdict(os.path.join(self.path + 'tb_dict_v2.txt'))
        self.stopwords = [line[0:-1] for line in
                          open(os.path.join(self.path + 'stopwords_new.txt'), 'r', encoding='utf-8').readlines()]  # 加载停用词词典

    def cal_tfidf(self, tag_all):
        vectorizer = CountVectorizer(stop_words=None)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(tag_all))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        return weight

    def calculate(self, cases=None, opcInfo=None, tags=None):
        if cases is None or len(cases) == 0:
            return [DiagnoseResponse()]
        else:
            # self.path=os.path.split(os.path.realpath(__file__))[0]

            #             stopwords=[line[0:-1] for line in open(self.path + '\\data\\stopwords_new.txt','r',encoding='utf-8').readlines()] #加载停用词词典
            fre_k = max(int(i.frequency) for i in cases) / 4  # 计算sigmoid函数的系数

            tags_input = ''.join(tags)
            input_cut = jieba.cut(tags_input)
            result = []
            for word in input_cut:
                if word not in self.stopwords:
                    result.append(word)

            tag_all = [' '.join(result)]

            #            tag_all=[' '.join(tags)]   #将输入的tag作为首行
            for case in cases:
                result = []
                seg_list = jieba.cut(case.casesName)
                for word in seg_list:
                    if word not in self.stopwords:
                        result.append(word)
                tags_build = ' '.join(result)
                tag_all.append(tags_build)
            if tags is None or len(tags) == 0:
                cases.sort(key=lambda x: int(x.frequency), reverse=True)
                dr_case = cases[0:5]
                dr = []
                for case_sort in dr_case:
                    dr.append(DiagnoseResponse(case_sort, 1.0 / (1 + np.exp(-float(case_sort.frequency) / fre_k))))
                return dr
            else:
                weight = self.cal_tfidf(tag_all)
                A = weight[0]  # A为输入的tag的权重分布
                rate = []
                fre_k = max(int(i.frequency) for i in cases) / 4
                # 输入与case库依次计算距离（余弦距离/欧式距离）
                for i in range(1, len(weight)):
                    B = weight[i]
                    dist_rate = 1 - np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                    if dist_rate < 0.05:
                        fre_rate = ((1.0 / (1 + np.exp(-(float(cases[i - 1].frequency) / fre_k)))) - 0.5) * 0.1 + 0.95
                        rate.append((1 - dist_rate) * fre_rate)
                    else:
                        fre_rate = ((1.0 / (1 + np.exp(-(float(cases[i - 1].frequency) / fre_k)))) - 0.5) * 0.4 + 0.8
                        rate.append((1 - dist_rate) * fre_rate)
                        # rate.append(1-dist_rate)
                rate = np.array(rate)
                dist_argsort = np.argsort(1 - rate)  # 根据距离排序索引
                dist_argsort_knn = dist_argsort[0:5]  # 筛选前5条案例
                dr = []
                for index in dist_argsort_knn:
                    dr.append(DiagnoseResponse(cases[index], rate[index]))
                return dr


class Feature(object):
    def __init__(self):
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.stopwords_path = self.path + '/data/stopwords_new.txt'  # 加载停止词
        self.a_stopwords = []
        self.f_stopwords = open(self.stopwords_path, 'r', encoding='utf-8')
        self.a_stopwords = self.f_stopwords.readlines()
        self.dict_path = self.path + '/data/tb_dict_v2.txt'  # 加载词典
        # self.dict_path= 'data/tb_dict_v2.txt'
        self.jieba = jieba
        self.jieba.load_userdict(self.dict_path)
        self.tongyici_path = self.path + '/data/tongyici.txt'  # 加载同义词
        self.a_tongyici = []
        self.f_tongyici = open(self.tongyici_path, 'r', encoding='utf-8')
        self.a_tongyici = self.f_tongyici.readlines()

        self.opcinfo_path = self.path + r'/data/opcInfo.txt'  # 加载opc信息
        self.a_opcinfo = []
        # self.aa_opcinfo=[]
        self.f_opcinfo = open(self.opcinfo_path, 'r', encoding='utf-8')
        self.a_opcinfo = self.f_opcinfo.readlines()
        # self.aa_opcinfo=self.f_opcinfo.readlines()

        self.component_path = self.path + '/data/component.txt'  # 加载部位名称
        self.a_component = []
        self.f_component = open(self.component_path, 'r', encoding='utf-8')
        self.a_component = self.f_component.readlines()

    def recognize(self, inputs=None):
        opcInfo, componentName, tag = self.inputs_check(inputs)
        # frr=FeatureRecognizeResponse('opcInfo','componentName',['tag1','tag2'])
        # frr=FeatureRecognizeResponse(opcInfo,componentName,tag)
        return [FeatureRecognizeResponse(opcInfo, componentName, tag)]

        # 定义停止词

    def stopwordslist(self):  # 定义停止词路径
        #        stopwords=[line.strip() for line in open(self.stopwords_path,'r',encoding='utf-8').readlines()] #加载停用词词典
        stopwords = [line.strip() for line in self.a_stopwords]  # 加载停用词词典
        return stopwords

    # 输入预处理，去掉标点符号
    def remove_symbol(self, string):  # 文件预处理，去掉标点符号
        return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，“”。？?、~@#￥%……&*（）]+", "", string)

    # 对inputs分词
    def seperate_word(self, string):  # 对输入的inputs分词
        self.jieba = jieba
        # sep_word=inputs.split('||',2)
        yu_inputs = self.remove_symbol(string)
        # jieba.load_userdict("D:/tb_dict_v2.txt") #加载分词词典
        # jieba.load_userdict(self.dict_path) #加载分词词典
        result = []
        seg_list = self.jieba.cut(yu_inputs)  # 对输入进行分词
        # stopwords=stopwordslist("D:/stopwords_new.txt") #加载停用词的路径
        stopwords = self.stopwordslist()  # 加载停用词的路径
        for w in seg_list:  # 去停用词
            if w not in stopwords:
                if w != '\t':
                    result.append(w)
        combine_dict = {}  # 合并同义词
        # for w in open("D:\\tongyici.txt",'r',encoding='utf-8'):
        for w in self.a_tongyici:
            seperate_word = w.encode('utf-8').decode('utf-8-sig').strip().split("\t")
            num = len(seperate_word)
            for i in range(1, num):
                combine_dict[seperate_word[i]] = seperate_word[0]
        seg_result = []  # 输出同义词的第一个词，并合并分词结果
        for word in result:
            if word in combine_dict:
                word = combine_dict[word]
                seg_result.append(word)
            else:
                seg_result.append(word)
        seg_result = np.unique(seg_result)  # 对切分的词去重处理
        return seg_result

        # 对inputs进行检索，返回opcInfo,componentName,tag

    def inputs_check(self, inputs=None):
        # self.val_list=val_list
        tag = []
        #        tag_ds=[]
        #        tag_DS=[]
        tag_opc = []
        tag_OPC = []
        check_opc = []
        word_com = []
        word_COM = []
        tag_COM = []
        # componentName=None
        # opcInfo=''
        # componentName_ds=[]
        componentName = ''
        sep_word = inputs.split('||', 2)  # 切分输入的组合字符串
        componentName1 = sep_word[0]  # 第一个元素为componentName
        tag_s1 = sep_word[1]  # 第二个元素为组合字符串型的tag数据
        description = sep_word[2]  # 第三个元素为描述性语句

        for line in self.a_opcinfo:
            opc_word = line.encode('utf-8').decode('utf-8-sig').strip('\n')
            tag_opc.append(opc_word)
        for line in tag_opc:
            opc_WORD = line.strip()
            tag_OPC.append(opc_WORD)

        for line in self.a_component:
            com_word = line.encode('utf-8').decode('utf-8-sig').strip('\n')
            word_com.append(com_word)
        for line in word_com:
            com_WORD = line.strip()
            word_COM.append(com_WORD)

        # 判断第一个切分元素componentName
        if componentName1 == ' ':
            #            componentName=''
            if description == ' ':
                componentName = ''
            else:
                check_opc = [line1 for line1 in tag_OPC if line1 in description]
                tag_check_opc = ''.join(check_opc)
                opcInfo = tag_check_opc
                tag_CHECK_OPC = description.strip(tag_check_opc)
                tag_check_OPC = self.remove_symbol(tag_CHECK_OPC)
                seg_result = self.seperate_word(tag_check_OPC)
                for w in seg_result:
                    tag_COM.append(w)

                combine_dict = {}  # 合并同义词
                for w in self.a_tongyici:
                    seperate_word = w.encode('utf-8').decode('utf-8-sig').strip().split("\t")
                    num = len(seperate_word)
                    for i in range(1, num):
                        combine_dict[seperate_word[i]] = seperate_word[0]
                tag_com = []
                for word in tag_COM:
                    if word in combine_dict:
                        word = combine_dict[word]
                        tag_com.append(word)
                    else:
                        tag_com.append(word)
                componentName_DS = []
                componentName_DS = [line1 for line1 in tag_com if line1 in word_COM]
                st = '||'
                componentName_DS = st.join(componentName_DS)
                componentName = componentName + '||' + componentName_DS  # 若出现不同的componentName，则连接起来
                sep_component = componentName.split('||')
                sep_component = np.unique(sep_component)  # 对componentName去重
                # st='||'
                componentName = st.join(sep_component)  # 将componentName连接成单个字符串
        else:
            if description == ' ':
                componentName = componentName1
                combine_dict = {}  # 合并同义词
                for w in self.a_tongyici:
                    seperate_word = w.encode('utf-8').decode('utf-8-sig').strip().split("\t")
                    num = len(seperate_word)
                    for i in range(1, num):
                        combine_dict[seperate_word[i]] = seperate_word[0]
                if componentName in combine_dict:
                    componentName = combine_dict[componentName]
                else:
                    componentName = componentName
            else:
                componentName = componentName1
                combine_dict = {}  # 合并同义词
                for w in self.a_tongyici:
                    seperate_word = w.encode('utf-8').decode('utf-8-sig').strip().split("\t")
                    num = len(seperate_word)
                    for i in range(1, num):
                        combine_dict[seperate_word[i]] = seperate_word[0]
                if componentName in combine_dict:
                    componentName = combine_dict[componentName]
                else:
                    componentName = componentName
                check_opc = [line1 for line1 in tag_OPC if line1 in description]
                tag_check_opc = ''.join(check_opc)
                opcInfo = tag_check_opc
                tag_CHECK_OPC = description.strip(tag_check_opc)
                tag_check_OPC = self.remove_symbol(tag_CHECK_OPC)
                seg_result = self.seperate_word(tag_check_OPC)
                for w in seg_result:
                    tag_COM.append(w)

                combine_dict = {}  # 合并同义词
                for w in self.a_tongyici:
                    seperate_word = w.encode('utf-8').decode('utf-8-sig').strip().split("\t")
                    num = len(seperate_word)
                    for i in range(1, num):
                        combine_dict[seperate_word[i]] = seperate_word[0]
                tag_com = []
                for word in tag_COM:
                    if word in combine_dict:
                        word = combine_dict[word]
                        tag_com.append(word)
                    else:
                        tag_com.append(word)
                componentName_DS = []
                componentName_DS = [line1 for line1 in tag_com if line1 in word_COM]
                st = '||'
                componentName_DS = st.join(componentName_DS)
                componentName = componentName + '||' + componentName_DS  # 若出现不同的componentName，则连接起来
                sep_component = componentName.split('||')
                sep_component = np.unique(sep_component)  # 对componentName去重
                # st='||'
                componentName = st.join(sep_component)  # 将componentName连接成单个字符串

        # 判断第二个切分元素tag
        if tag_s1 == ' ':
            # tag=None
            tag = []
        else:
            tag = tag_s1.split(',')  # 得到list型的tag数据

        # 判断第三个切分元素OPC+tag
        if description == ' ':
            opcInfo = None
        else:
            check_opc = [line1 for line1 in tag_OPC if line1 in description]
            tag_check_opc = ''.join(check_opc)
            if tag_check_opc == '':  # 无OPC信息，仅对description分词
                seg_result = self.seperate_word(description)
                for w in seg_result:
                    tag.append(w)
            else:
                opcInfo = tag_check_opc
                tag_CHECK_OPC = description.strip(tag_check_opc)
                tag_check_OPC = self.remove_symbol(tag_CHECK_OPC)
                seg_result = self.seperate_word(tag_check_OPC)
                for w in seg_result:
                    tag.append(w)
        componentName = componentName.lstrip("||")
        return opcInfo, componentName, tag


class Diagnose(object):

    def __init__(self):
        self.ocr_model = OCR()
        self.similarity_calculator = Similarity()
        self.feature_recognizer = Feature()

    def diagnose(self, cases, opcInfo, tags):
        """
        Parameters:
         - cases
         - opcInfo
         - tags
        """
        return self.similarity_calculator.calculate(cases, opcInfo, tags)

    def ocr(self, image):
        """
        Parameters:
         - image
        """
        return self.ocr_model.recognize(image)

    def feature_recognize(self, message):
        """
        Parameters:
         - message
        """
        return self.feature_recognizer.recognize(message)