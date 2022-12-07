import re
import random
import math
import pandas as pd

from LSTM_model.lstm_model import *

number_population = 200# 种群数量
Number_of_iterations = 200#迭代数
population = []#种群
probability_cross = 0.6#交叉概率
probability_variation = 0.2#变异概率
M100 = '100'
m10 = '10'
dict = {

    # 这里的信息由具体要优化的的超参数决定

    # 'head_nl': 'hnl', 'tail_nl': 'nlt',
    'head_nn': 'hnn', 'tail_nn': 'nnt',
    'head_af': 'haf', 'tail_af': 'aft',
    'head_dr': 'hdr', 'tail_dr': 'drt',
    'head_hs': 'hhs', 'tail_hs': 'hst',
    'head_op': 'hop', 'tail_op': 'opt',
    # 'head_lr': 'hlr', 'tail_lr': 'lrt',
    'head_mb': 'hmb', 'tail_mb': 'mbt'
}
dict_cols = list(dict.keys())
def take_head():
    parameter_head = []
    for i in range(0,len(dict_cols),2):
        parameter_head.append(dict_cols[i])
    return parameter_head
parameter_head = take_head()
# print(parameter_head)
Function_character = ['+', '-', '*', '/']
Fun_cha_triangle = ['S','C']
gene_tails = []
for i_t in range(1,10):
    gene_tails.append(str(i_t))
# print(gene_tails)

def getfun_cha(n):
    fun = ''
    for i in range(n):
        fun = fun + Function_character[random.randint(0,3)]

    return fun

def gettail(n):
    tail = ''
    for i in range(n):
        tail = tail + gene_tails[random.randint(0,8)]
    return  tail

# 1~9
def generate_n_layers():
    gene = ''
    gene = dict['head_nl'] + 'I+' + '%' + getfun_cha(2) + gettail(3) + 'm1'+dict['tail_nl']
    # print('n_layers',gene)
    return gene

# 0~100
def generate_n_neurouse():
    gene = ''
    gene = dict['head_nn']+'I+'+'%'+getfun_cha(2)+gettail(3)+'M1'+dict['tail_nn']
    # print('n_neurouse',gene)
    return gene

# 0~1
def generate_drop_rate():
    gene = ''
    gene = dict['head_dr']+'A'+Fun_cha_triangle[random.randint(0,1)]+getfun_cha(2)+gettail(3)+dict['tail_dr']
    # print('drop_rate',gene)
    return gene

# 1~100
def generate_hidden_size():
    gene = ''
    gene = dict['head_hs'] + 'I+' + '%' + getfun_cha(2) + gettail(3) + 'M1'+dict['tail_hs']
    # print('hidden_size',gene)
    return gene

# 2^1~9
def generate_Mini_batch():
    gene = ''
    gene = dict['head_mb'] + 'IP' +'2'+ gettail(1)+dict['tail_mb']
    # print('Mini_batch',gene)
    return gene

# 0,1,2,3
def gennerate_Active_function():
    gene = ''
    gene = dict['head_af'] + 'I%' + getfun_cha(2) + gettail(3) + '4'+dict['tail_af']
    # print('Active_function',gene)
    return gene

# 0,1,2,3
def generate_Optimizer():
    gene = ''
    gene = dict['head_op'] + 'I%' + getfun_cha(2) + gettail(3) + '4'+dict['tail_op']
    # print('Optimizer',gene)
    return gene

# 0.00001~0.01
def gennerate_Learning_rate():
    gene = ''
    gene = dict['head_lr'] + '!P' + 'm'+str(random.randint(2,5))+ gettail(1) +dict['tail_lr']
    # print('Learning_rate',gene)
    return gene
# generate_n_layers()
# generate_n_neurouse()
# generate_drop_rate()
# generate_hidden_size()
# generate_Mini_batch()
# gennerate_Active_function()
# generate_Optimizer()
# gennerate_Learning_rate()
# print(2**8)

# 初始化
def initial_population():
    individual = ''
    for i in range(number_population):
        # generate_n_layers()+\    gennerate_Learning_rate()+\
        individual = \
                     generate_n_neurouse()+ \
                     gennerate_Active_function() + \
                     generate_drop_rate()+\
                     generate_hidden_size()+ \
                     generate_Optimizer() + \
                     generate_Mini_batch()
        population.append(individual)
    # return population

# initial_population()
# print(population)

# 处理染色体
def deal_chromosome(chrome):
    this_dict = {}
    # this_dict['nl'] = re.findall(r'{}(.+){}'.format(dict['head_nl'],dict['tail_nl']), chrome)[0]
    this_dict['nn'] = re.findall(r'{}(.+){}'.format(dict['head_nn'], dict['tail_nn']), chrome)[0]
    this_dict['af'] = re.findall(r'{}(.+){}'.format(dict['head_af'], dict['tail_af']), chrome)[0]
    this_dict['dr'] = re.findall(r'{}(.+){}'.format(dict['head_dr'], dict['tail_dr']), chrome)[0]
    this_dict['hs'] = re.findall(r'{}(.+){}'.format(dict['head_hs'], dict['tail_hs']), chrome)[0]
    this_dict['op'] = re.findall(r'{}(.+){}'.format(dict['head_op'], dict['tail_op']), chrome)[0]
    # this_dict['lr'] = re.findall(r'{}(.+){}'.format(dict['head_lr'], dict['tail_lr']), chrome)[0]
    this_dict['mb'] = re.findall(r'{}(.+){}'.format(dict['head_mb'], dict['tail_mb']), chrome)[0]
    # print('========',this_dict)
    return  this_dict

# ceshi = deal_chromosome(population[0])

# 解码_计算
def To_calculate(functor, fig):
    re1, re2= None, None
    fun = None
    while(len(functor)>0):
        result = 0
        fun = functor.pop()
        if(fun=='P'):
            if(re1 == None):
                re1 = float(fig.pop())
            if(re2 == None):
                re2 =float(fig.pop())
            # print('re1**re2', re1, re2 )
            re1 = re1**re2

            re2 = None

        if (fun == '/'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1/re2', re1, re2 )
            re1 = re1 / re2

            re2 = None

        if (fun == '+'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1+re2', re1, re2)
            re1 = re1 + re2
            re2 = None

        if (fun == '-'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1-re2', re1, re2)
            re1 = re1 - re2
            re2 = None

        if (fun == '*'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1*re2', re1, re2)
            re1 = re1 * re2
            re2 = None

        if (fun == '!'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1!re2', re1, re2)
            re1 = re2 / re1
            re2 = None

        if (fun == '%'):
            if (re1 == None):
                re1 = float(fig.pop())
            if (re2 == None):
                re2 = float(fig.pop())
            # print('re1%re2', re1, re2)
            re1 = re1 % re2
            re2 = None

        if (fun == 'S'):
            if (re1 == None):
                re1 = float(fig.pop())
            # print('sin(re1)', re1, re2)
            re1 = math.sin(re1)

        if (fun == 'C'):
            if (re1 == None):
                re1 = float(fig.pop())
            # print('cos(re1)', re1, re2)
            re1 = math.cos(re1)

        if (fun == 'A'):
            if (re1 == None):
                re1 = float(fig.pop())
            # print('abs(re1)', re1, re2)
            re1 = abs(re1)

        if (fun == 'I'):
            if (re1 == None):
                re1 = float(fig.pop())
            # print('int(re1)', re1, re2)
            re1 = int(re1)

    return re1


# 解码
def decod(gene):
    decod_dict = gene
    decod_dict_cols = list(decod_dict.keys())
    # print(decod_dict)
    # print(len(decod_dict_cols))
    for i in range(len(decod_dict_cols)):
        str1 = decod_dict[decod_dict_cols[i]]
        functor, fig = [], []
        for j in range(len(str1)):
            ind = str1[j]
            if(ind=='m' or ind=='M' or ind.isdigit()):
                if(ind =='m'):
                    ind = m10
                if(ind =='M'):
                    ind = M100
                fig.append(ind)
            else:
                functor.append(ind)
        fig.reverse()#反序
        # print(decod_dict_cols[i],functor, fig)
        result = To_calculate(functor, fig)
        decod_dict[decod_dict_cols[i]] = result
    return decod_dict

# 计算适应度并且排序
def find_fitness():
    fitness = []
    for i in range(number_population):
    # for i in range(10):
        dict_para = decod(deal_chromosome(population[i]))
        # print(dict_para)
        # n_layers = dict_para['nl']
        n_neurouse = dict_para['nn']
        Active_function = dict_para['af']
        drop_rate = round(dict_para['dr'],2)
        hidden_size = dict_para['hs']
        Optimizer = dict_para['op']
        # Learning_rate = dict_para['lr']
        Mini_batch = dict_para['mb']
        mse = train_net(n_neurouse, drop_rate,
                        hidden_size, Mini_batch, Active_function, Optimizer, )
        fitness.append((mse))
    dict_fitness = {
        'fitness':fitness,
        'population':population
    }
    df_fitness = pd.DataFrame(dict_fitness)
    df_fitness = df_fitness.sort_values(by='fitness')
    df_fitness = df_fitness.reset_index(drop=True)
    return df_fitness
# fitness = find_fitness(population)
# print(fitness)

# 轮盘选择
def roulette_choose(fits):
    choose_population = []
    while(len(choose_population)<number_population):
        index = random.randint(1,number_population)
        for i in range(index):
            choose_population.append(fits.population[i])
            if(len(choose_population)==number_population):
                return choose_population
    return choose_population



# 交换列表两个元素位置
def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

# 随机选择交叉的两个基因
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
    cross_gene[1] = re.sub(r'head_',"tail_",cross_gene[1]) #在后面的
    # print(index_gene1, index_gene2)
    # print(gene1, gene2)
    # print(cross_gene)
    return cross_gene

# 交叉
def cross():
    for i in range(number_population):
        probability = random.random()
        if(probability<probability_cross):
            other_ch = random.randint(0,number_population-1)
            other_chromosome = population[other_ch]
            cross_gene = chose_twogene()
            # print(i)
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
            # print('==========================================')
            # print('this_chromosome:',population[i])
            # print('other_chromosome:',other_chromosome)
    # return population
# cross(population)

# 根据变异位置返回变异结果
def variation_g(parameter):

    if(parameter=='head_nl'):
        return generate_n_layers()
    if(parameter=='head_nn'):
            return generate_n_neurouse()
    if(parameter=='head_af'):
            return gennerate_Active_function()
    if(parameter=='head_dr'):
            return generate_drop_rate()
    if(parameter=='head_hs'):
            return generate_hidden_size()
    if(parameter=='head_op'):
            return generate_Optimizer()
    if(parameter=='head_lr'):
            return gennerate_Learning_rate()
    if(parameter=='head_mb'):
            return generate_Mini_batch()


# 变异
def variation():
    for i in range(number_population):
        probability = random.random()
        if (probability<probability_variation):
            veriation_gene = (random.sample(parameter_head, 1))
            this_parameter_head = re.findall(r'_.+', veriation_gene[0])
            veriation_gene.append('tail'+this_parameter_head[0])
            veriation_str = re.findall(r'{}.+{}'.format(dict[veriation_gene[0]],dict[veriation_gene[1]]), population[i])
            new_veriation_str = variation_g(veriation_gene[0])
            # print(i)
            # print('veriation_gene:',veriation_gene)
            # print('oldpopolation:',population[i])
            population[i] = re.sub(r'{}.+{}'.format(dict[veriation_gene[0]], dict[veriation_gene[1]]), new_veriation_str,population[i])
            # print('newpopolation:', population[i])
            # print('===================================')

# variation(population)
# print(len(population))


if __name__ == '__main__':
    best_fitness_change = []
    initial_population()
    print('loading.....running_lstm------item:0')
    fitness = find_fitness()
    dict_finess = {
        'fitness':fitness['fitness'][0],
        'population':fitness['population'][0],
        'item': 0
    }
    best_fitness_change.append(dict_finess)
    for item in range(1,Number_of_iterations):
        print("\n\n")
        population = roulette_choose(fitness)
        cross()
        variation()
        print('loading.....running_lstm------item:{}'.format(item))
        fitness = find_fitness()
        if(fitness['fitness'][0]<best_fitness_change[-1]['fitness']):
            dict_finess = {
                'fitness': fitness['fitness'][0],
                'population': fitness['population'][0],
                'item': item
            }
            best_fitness_change.append(dict_finess)
        print('At present best is:{}'.format(best_fitness_change[-1]))
        print('population:{}'.format(decod(deal_chromosome(best_fitness_change[-1]['population']))))
    # print('fitness:',fitness)
    # print('population:',population)

