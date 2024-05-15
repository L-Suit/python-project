def fun1():
    """
    这是一个输出函数，输出一个语句
    :return:void
    """
    print("shuchu test1")
    print('函数调用完毕')

def add(a,b):
    '''
    加法函数，
    :param a:加数 1
    :param b: 加数2
    :return: 相加的结果
    '''
    c=a+b
    return c

ans = add(12,55)
print(ans)