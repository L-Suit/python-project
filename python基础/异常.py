num = input("请输入运算的数字")

try:
    ans = 20/int(num)
    print('计算结果是：',ans)
except ZeroDivisionError:
    print('你输入的数字有误')


