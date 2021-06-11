
import numpy as np
import random
import math


def GenerateData(mu, var, alpha):
    """
    生成数据
    :param mu: 均值可以是单个数据，也可以是向量
    :param var: 标准差
    :param alpha: 权重
    :return: 3个高斯混合模型的数据集
    """

    SampleNum = 100  # 数据长度
    ModelNum = 3  # 模型个数
    DataSample = []  # 存放数据list

    for i in range(ModelNum):
        temp = np.random.normal(mu[i], var[i], int(SampleNum * alpha[i]))
        DataSample.extend(temp)  # 每次循环加入数据

    random.shuffle(DataSample)  # 打乱样本

    # 返回数据集，数据类型list，数据长度100*1
    return DataSample


def Gauss(DataSampleArr, mu, var):
    '''
    高斯公式函数
    :param DataSampleArr: 数据集 array数据格式
    :param mu: 均值可以是单个数据，也可以是向量
    :param var: 标准差
    :return: 随机变量的pdf
    '''

    pdf = (1 / (math.sqrt(2 * math.pi) * var ** 2)) * np.exp(
        -1 * (DataSampleArr - mu) * (DataSampleArr - mu) / (2 * var ** 2))

    return pdf


def E_step(DataSampleArr, alpha, mu, var):
    '''
    EM算法中的E步 计算分模型k对观数据y的响应度
    :param DataSampleArr: 可观测数据
    :param alpha: 权重系数
    :param mu: 均值
    :param var: 标准差
    :return: 两个模型各自的响应度gamma
    '''
    # 计算响应度
    # 先计算模型0的响应度的分子
    gamma = [[0] * 100 for _ in range(3)]
    # 三个模型的响应度
    for i in range(3):
        gamma[i] = alpha[i] * Gauss(DataSampleArr, mu[i], var[i])

    # 相加为总的分布
    sum = gamma[0] + gamma[1] + gamma[2]

    # 各自相除，得到各自模型的响应度
    for i in range(3):
        gamma[i] = gamma[i] / sum

    # 更新模型响应度
    return gamma


def M_step(gamma, mu, DataSampleArr):
    '''
    M步 计算均值，标准差，权重
    :param DataSampleArr: 可观测数据

    :param mu: 均值
    :param gamma: 响应度
    :return: 均值，标准差，权重
    '''

    # np.dot 点积,sum(一个矩阵)是总的求和，到最低维度
    # 均值的估计值
    mu_new = [0] * 3
    for i in range(3):
        mu_new[i] = np.dot(gamma[i], DataSampleArr) / np.sum(gamma[i])

    var_new = [0] * 3
    # math.sqrt  平方根
    # 标准差的估计值
    for i in range(3):
        var_new[i] = math.sqrt(np.dot(gamma[i], (DataSampleArr - mu[i]) ** 2) / np.sum(gamma[i]))

    # 权重系数的估计值
    alpha_new = [0] * 3
    for i in range(3):
        alpha_new[i] = np.sum(gamma[i]) / len(gamma[i])

    # 将更新的值返回
    return alpha_new, mu_new, var_new


def EM_Train(DataSampleList, iter=2000):
    '''
    根据EM算法进行参数估计
    :param DataSampleList:数据集
    :param iter: 迭代次数
    :return: 估计的参数
    '''
    # 将可观测数据y转换为数组形式，主要是为了方便后续运算
    DataSampleArr = np.array(DataSampleList)

    # 步骤1：对参数取初值，开始迭代

    alpha = [0.2, 0.45, 0.35]
    mu = [-3, 4, 2]
    var = [0.4, 0.9, 0.8]

    # 开始迭代
    step = 0
    while (step < iter):
        # 每次进入一次迭代后迭代次数加1
        step += 1
        # 步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度
        gamma = E_step(DataSampleArr, alpha, mu, var)
        # 步骤3：M步
        alpha, mu, var = M_step(gamma, mu, DataSampleArr)  # 注意保证M返回的参数名字和E一致，否则进不去迭代
    # 迭代结束后将更新后的各参数返回
    return alpha, mu, var


if __name__ == '__main__':
    mu = [-2, 5, 1]
    var = [0.5, 1, 1]
    alpha = [0.1, 0.6, 0.3]
    # 生成数据
    DataSampleList = GenerateData(mu, var, alpha)

    # 打印参数真实值
    print('the parameters are:')
    print('alpha,mu,var:')
    print(alpha, mu, var)

    alpha, mu, var = EM_Train(DataSampleList)

    # 保留小数
    format_alpha = [round(i, 3) for i in alpha]
    format_mu = [round(i, 3) for i in mu]
    format_var = [round(i, 3) for i in var]
    # 打印估计值
    print('the predictations are:')
    print('alpha,mu,var:')
    print(format_alpha, format_mu, format_var)

