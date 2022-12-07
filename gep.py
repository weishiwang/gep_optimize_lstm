import re
import random
import math
#
# str = 'hqiu11111111fqiuhwei10000000fwei'
# ress = re.findall(r"hqiu(.*)fqiu", str)
# print(type(ress))
# print(ress)
# notnumber =re.finditer(r"\D+", str)
# # print(notnumber)
# for it in notnumber:
#     print(it.group())
# 进制转换
# a = 10
# b = format(int(a), '08b')
# print(b)
# print(int('1111111111111111',2))

# 测试字典的使用
# dict = {
#     'head_nl': 'hnl', 'tail_nl': 'tnl',
#     'head_nn': 'hnn', 'tail_nn': 'tnn',
#     'head_af': 'haf', 'tail_af': 'haf',
#     'head_dr': 'hdr', 'tail_dr': 'tdr',
#     'head_hs': 'hhs', 'tail_hs': 'ths',
#     'head_op': 'hop', 'tail_op': 'top',
#     'head_lr': 'hlr', 'tail_lr': 'tlr',
#     'head_mb': 'hmb', 'tail_mb': 'tmb'
#         }
# dict_cols = list(dict.keys())
# print(type(dict_cols))
# for i in range(0,len(dict_cols),2):

    # print(i)
    # print(dict_cols[i])

# print(dict_cols)
# for item in len(dict):
#     print(item)
#     print(dict[item])
# 查找字符串位置

# str1="ABAAABCDBBABCDDEBCABCDDDABCJJJKK"
# pattern="ABC"
#
# def pattern_search1(string,pattern):
#     index=0
#     while index<len(string)-len(pattern):
#         index=string.find(pattern,index,len(string))
#         if index==-1:
#             break
#         yield index
#         index+=len(pattern)-1

# print(list(pattern_search1(str1,pattern)))

# 头部固定信息
dict = {

    # 这里的信息由具体要优化的的超参数决定

    'head_nl': 'hnl', 'tail_nl': 'nlt',
    'head_nn': 'hnn', 'tail_nn': 'nnt',
    'head_af': 'haf', 'tail_af': 'aft',
    'head_dr': 'hdr', 'tail_dr': 'drt',
    'head_hs': 'hhs', 'tail_hs': 'hst',
    'head_op': 'hop', 'tail_op': 'opt',
    'head_lr': 'hlr', 'tail_lr': 'lrt',
    'head_mb': 'hmb', 'tail_mb': 'mbt'
        }
dict_cols = list(dict.keys())
number_population = 200# 种群数量
population = []#种群
probability_cross = 0.6#交叉概率
probability_variation = 0.2#变异概率
# 返回基因头和尾
# def return_ht(n):

# 交换列表两个元素位置
def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list
# 将头部基因名取出作为标识
def take_head():
    parameter_head = []
    for i in range(0,len(dict_cols),2):
        parameter_head.append(dict_cols[i])
    return parameter_head
parameter_head = take_head()
print(parameter_head)
# 生成二进制随机数
def binary():
    rand = random.randint(0, 100)
    # print(format(int(rand), '08b'))
    return format(int(rand), '08b')
# binary()
# 初始化种群
def initial_population():
    '''
    Number of full connected layers
    Number of neurons
    Active  function
    Dropout rate
    Hidden size
    Optimizer
    Learning rate
    Mini - batch
    '''

    # head_nl, tail_nl = 'hnl','tnl'
    # head_nn, tail_nn = 'hnn', 'tnn'
    # head_af, tail_af = 'haf', 'haf'
    # head_dr, tail_dr = 'hdr', 'tdr'
    # head_hs, tail_hs = 'hhs', 'ths'
    # head_op, tail_op = 'hop', 'top'
    # head_lr, tail_lr = 'hlr', 'tlr'
    # head_mb, tail_mb = 'hmb', 'tmb'
    # dict = {
    #     'head_nl': 'hnl', 'tail_nl': 'tnl',
    #     'head_nn': 'hnn', 'tail_nn': 'tnn',
    #     'head_af': 'haf', 'tail_af': 'haf',
    #     'head_dr': 'hdr', 'tail_dr': 'tdr',
    #     'head_hs': 'hhs', 'tail_hs': 'ths',
    #     'head_op': 'hop', 'tail_op': 'top',
    #     'head_lr': 'hlr', 'tail_lr': 'tlr',
    #     'head_mb': 'hmb', 'tail_mb': 'tmb'
    # }
    # n_individual =
    for j in range(number_population):
        chromosome = ''
        for i in range(0,len(dict_cols),2):
            chromosome = chromosome + dict[dict_cols[i]] + binary() + dict[dict_cols[i+1]]
        population.append(chromosome)
        # print(chromosome)
    # print(population,len(population))
    return population
# initial_population()
population = initial_population()
# 处理染色体
def deal_chromosome(chromosome):
    # pattern = re.compile(r'h[a-z]+\d+[a-z]+t')  # 查找数字
    pattern = re.compile(r'\d+')  # 查找数字
    result1 = pattern.findall(chromosome)
    # result2 = pattern.findall('run88oob123google456', 0, 10)
    # 这里还需要再加一个处理二进制对应的超参数，具体情况在具体分析
    print(result1)
deal_chromosome(initial_population()[1])

# 现在还只能交叉变异

def chose_twogene():

    cross_gene = random.sample(parameter_head, 2)  # 随机选择要交叉的两个基因
    # print(cross_gene)
    gene1 = cross_gene[0]
    gene2 = cross_gene[1]
    index_gene1 = parameter_head.index(gene1)
    index_gene2 = parameter_head.index(gene2)
    while(math.fabs(index_gene1-index_gene2)==len(parameter_head)-1):
        cross_gene = random.sample(parameter_head, 2)  # 随机选择要交叉的两个基因
        # print(cross_gene)

        gene1 = cross_gene[0]
        gene2 = cross_gene[1]
        index_gene1 = parameter_head.index(gene1)
        index_gene2 = parameter_head.index(gene2)
    if (index_gene1 > index_gene2):#排序
        cross_gene = swapPositions(cross_gene, 0, 1)
    cross_gene[1] = re.sub(r'head_',"tail_",cross_gene[1])
    # print(index_gene1, index_gene2)
    # print(gene1, gene2)
    # print(cross_gene)
    return cross_gene

# 交叉
def cross(population):
    for i in range(number_population):
        probability = random.random()
        if(probability<probability_cross):
            other_ch = random.randint(0,number_population)
            other_chromosome = population[other_ch]
            cross_gene = chose_twogene()
            print(i)
            # print(cross_gene)
            # print(dict[cross_gene[0]])
            # print(type(dict[cross_gene[0]]))
            # print('this_chromosome',population[i])
            # print('other_chromosome', population[other_ch])
            this_tab = re.findall(r'{}.+{}'.format(dict[cross_gene[0]], dict[cross_gene[1]]), population[i])
            other_tab = re.findall(r'{}.+{}'.format(dict[cross_gene[0]], dict[cross_gene[1]]), population[other_ch])
            # print('this_tab',this_tab)
            # print('other_tab',other_tab)
            population[i] = re.sub(r'{}.+{}'.format(dict[cross_gene[0]], dict[cross_gene[1]]), other_tab[0], population[i])
            population[other_ch] = re.sub(r'{}.+{}'.format(dict[cross_gene[0]], dict[cross_gene[1]]), this_tab[0], population[other_ch])
            # print('this_chromosome',population[i])
            # print('other_chromosome', population[other_ch])
            print('==========================================')
            # print('this_chromosome:',population[i])
            # print('other_chromosome:',other_chromosome)
    return population

# cross(population)

# 变异
def variation(population):
    for i in range(number_population):
        probability = random.random()
        if (probability<probability_variation):
            veriation_gene = (random.sample(parameter_head, 1))
            this_parameter_head = re.findall(r'_.+', veriation_gene[0])
            veriation_gene.append('tail'+this_parameter_head[0])
            veriation_str = re.findall(r'{}.+{}'.format(dict[veriation_gene[0]],dict[veriation_gene[1]]), population[i])
            # 变异
            print(i)
            print(' veriation_str', veriation_str)
            print('old population[i]',population[i])
            veriation_str = re.sub(r'\d+', binary(), veriation_str[0])
            population[i] = re.sub(r'{}.+{}'.format(dict[veriation_gene[0]], dict[veriation_gene[1]]), veriation_str, population[i])
            print('already variation population[i]', population[i])
            print(' veriation_str', veriation_str)
            print(type(veriation_gene))
            print('this_parameter_head', this_parameter_head)
            print('veriation_gene', veriation_gene)
    return population
variation(population)
# 列表中随机取出两个元素？
# mylist = ['a','b','c','d']

# print(random.sample(parameter_head, 2))
# str1 = '123abcd43434344tt678ooo90'
# str2 = 'abcd'
# str4 = 'tt'
# str3 = '我是替换'
# str1 = re.sub(r"3{}.+4{}".format(str2,str4),str3,str1)
# print(str1)
