import csv
import heapq
import pandas as pd
from math import sqrt
from math import pow
import random
from scipy.interpolate import interp1d
import numpy as np
import pkg as pkg
import seaborn as seaborn
import sns
from matplotlib import pyplot as plt
import pymysql


import select_edge


def select_testdata(n,m,j,test_data):#取测试数据
    test_data1=[]
    conn = pymysql.connect(host='localhost', user='root', password='123456', db='test', port=3306, charset='utf8')
    cur = conn.cursor()
    # number=random.randint(1,31)
    number = j   #哪次数据
    n = str(n)   #哪个RP
    m = str(m)   #哪条边
    cur.execute('select length,rss,rss_vary from length_rss' + n + '_' + m + ' where number=%s', number)
    res = cur.fetchall()
    # print(res)
    rss = 0
    rss_vary = []
    for i in range(0, len(res)):
        rss = rss + float(res[i][1])
        rss_vary.append(float(res[i][2]))
    mean = round((rss / len(res)), 2)
    test_data1.append((mean,rss_vary))
    cur.close()
    conn.close()

    return test_data1


def result_RP(m,mean,rss_vary):
    result=[]
    result1 = []
    count = 0
    count0 = 0
    total = 0
    count1 = []
    count2 = []
    min_3 = []
    min_3index = []

    weight = []
    for n in range(1, 43):
        if n ==8 or n==13 or n==23 or n==2 or n==30:
            result.append(100000)
            continue

        x1 = []
        y1 = []
        n = str(n)
        m = str(m)  # 哪条边
        s = 'length_rss' + n + '_' + m + ''
        conn = pymysql.connect(host='localhost', user='root', password='123456', db='test', port=3306, charset='utf8')
        cur = conn.cursor()
        cur.execute(
            'select mean_total,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25 from 00ap_position '
            'where ap_position= %s', s)
        res = cur.fetchone()
        cur.close()
        conn.close()
        # print(res)
        mean_total=float(res[0])
        fingerprint=[]
        for i in range(1,len(res)):
            fingerprint.append(float(res[i]))
        # a = 3.0 / 17500.0
        # b = 32.0/175.0
        # # w1=0.4
        a1=abs(mean-mean_total)
        # if a1<=0:
        #     w1=0
        # else:
        #     x = a1
        #     w1 = a * x * x + b
        # ///////////////////////////////////////////////////////
        if a1>=10:
            a=1000

        else:
            w1=0.4
            w2 = 1-w1
            c=0
            for i in range(0,len(fingerprint)):
                b=rss_vary[i]-fingerprint[i]
                c=pow(b,2)+c
            a2=sqrt(c)
            a=w1*a1+w2*a2    #上面是求一个随RSS绝对值变大而变大的权重W1
        result.append(a)
    for i in range(0,len(result)):
        a=1.0/result[i]
        result1.append(a)#求欧氏距离


    x1=[]
    y1=[]
    TH_window=[]
    # print(result)
    result_ordered=[]
    result_ordered0=sorted(range(len(result1)),key=lambda k:result[k])  #按照从小到大排序RP
    result_ordered1=sorted(result1,reverse=True)  #按照从小到大排序求欧氏距离的倒数  在这里是每个RP的概率
    sum=0
    for i in range(0,len(result_ordered1)):
        sum=pow(result_ordered1[i],3)+sum

    for i in range(0,len(result1)):
        result_ordered.append((result_ordered0[i]+1,pow(result_ordered1[i],3)/sum))



    for i in range(0,len(result_ordered)):
        x1.append(result_ordered[i][0])
        y1.append(result_ordered[i][1])
        # if result_ordered[i][1]>TH:
        #     TH_window.append(result_ordered[i])
        #     TH_number=TH_number+1
    # print(TH_number,TH_window)
    dic=dict(zip(x1,y1))
    # print(dic)
    # plt.plot(x1, y1, 'g*--', alpha=0.5, linewidth=1)
    # for i in range(0,5):
    #     max_5=max_5+float(result_ordered[i][1])#最大的五个概率相加
    # # print(max_5)
    # for i in range(0, 3):
    #     max_3 = max_3 + float(result_ordered[i][1])  # 最大的五个概率相加
    # # print(max_3)
    # min_index=result.index(min(result))
    # # print(k,min_index+1)
    # min_3 = heapq.nsmallest(3, result)  # 最小的三个数据
    # min_3index = list(map(result.index, heapq.nsmallest(3, result)))  # 最小三个数据的索引
    # if k !=(min_index+1):
    #     count=count+1
    #     count1.append((k,min_index,result[min_index]))
    #     if k in min_3index:
    #         count0=count0+1
    #         count2.append((min_3index,min_3))
    weight.append(dic)
    return weight
def result_RP1(weight):
    for key,value in weight[0].items():
        if key in weight[1]:
            weight[1][key]+=value
            weight[1][key]=weight[1][key]
    weight=dict(sorted(weight[1].items(), key=lambda d:d[1],reverse=True))
    first_three_values=list(weight.values())[:3]
    # print(weight)
    sum_three=0
    for i in range(0,len(first_three_values)):
        sum_three+=first_three_values[i]
    print(sum_three)
    return weight
    # print(len(weight))
    # for i  in range(0,2):
    #     for j in range(0,23):
def Calculate_coordinates(test_xy,RP_xy,result,kn):#计算定位的坐标通过RP的概率
    p=0
    x=0
    y=0
    for i in range(0,kn):#len(result)
        p = result[i][1]+p

    for i in range(0, kn):
        x = x+RP_xy[result[i][0]][0] * result[i][1] / p
        y = y+RP_xy[result[i][0]][1] * result[i][1] / p
    error=sqrt(pow(x-test_xy[0],2)+pow(y-test_xy[1],2))
    print(error)


    return error




if __name__ == '__main__':
    # edge_data = [13,14,15,16,17]
    test_data = [1,2,3,4,5]    #,3,12,13,22,23]
    # test_data1=[26,27,28,29,30]
    RP_xy=[(0,0),(2.5,2.5),(10.3,4.9),(11,-4),(4,-4.2),(9.4,0),(5,-0.4),(-2.4,5),(8,0),(0.5,2.6),(1,-1.2),(-0.2,-1.5),(-0.2,-4.5),(-2,-6),(5.8,-5.8),
           (14,-5.8),(10,-8),(1.6,-7.8),(-0.2,-8.4),(2.4,4.6),(0.7,8.6),(4.4,8.2),(4.9,11),(3,14),(4.6,17.4),(-4,1),(-5,8),(-7,4),(-10,4),(-11,8),(0,0)
        ,(-4,-9),(-12,-9),(-4,-12),(-12,-12),(-12,-15),(-4,-15),(9,-10),(2,-10),(9,-13),(2,-12),(9,-16),(2,-16)]
    # RP_edge=[[(1,7,9)],(2),(3,6),(4),(5),()]
    # Node_data = [(1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (3, 5), (1, 6), (6, 7), (1, 7), (6, 8), (8, 9), (9, 10),
    #         (6, 10), (8, 10), (7, 11), (11, 12), (7, 12)]  # 索引为边序号
    # Edge_data = [(0), (1, 5, 7, 9), (1, 2), (2, 3, 6), (3, 4), (4, 5, 6), (7, 8, 10, 13), (8, 9, 15, 17), (10, 11), (11,12),
    #         (12, 13), (15, 16), (16, 17)]  # 索引为点序号,
    Node_data = [(1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (3, 5), (1, 6), (6, 7), (1, 7), (6, 8), (8, 9), (9, 10),
            (6, 10), (8, 10), (7, 11), (11, 12), (7, 12), (12, 13), (9, 15),(11, 14)]  # 索引为边序号
    Edge_data = [(0), (1, 5, 7, 9), (1, 2), (2, 3, 6), (3, 4), (4, 5, 6), (7, 8, 10, 13), (8, 9, 15, 17), (10, 11,14),
            (11, 12, 19),
            (12, 13, 14), (15, 16, 20), (16, 17, 18), (16, 17,18), (15, 16,20), (11, 12,19)]  # 索引为点序号
    RP_data70=[15, 16, 24, 26, 29, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42]
    RP_data60=[14, 17, 18, 21, 22, 27, 28, 38]
    RP_data50=[1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 19, 20, 25]
    RP_test=[]
    RP_error5=[]
    times=0
    times1=0
    x_three=[]
    for e in range(2,3):

        TH = e
        for kn in range(3,4):
        # if e ==4 :
        #     continue
            knn=3
            for r in range(0,2):
                random_select = r
                if r==0:
                    ALGO='Algo'
                else:
                    ALGO='Random'
                error = []
                len1=[]
                T1=[]
                T2=[]
                T3=[]
                T4 = []
                T5 = []
                T6 = []

                for m in range(1,21):
                    for k in range(1, 43):
                        # if k not in RP_data50:#
                        #     continue
                        if k == 8 or k == 13 or k==2 or k==23 or k==30:
                            continue
                        for j in test_data:
                            for ab in range(0,2):
                                begin_Node = Node_data[m][ab]

                                begin_edge = m
                                print(begin_edge,begin_Node )
                                # TH=3
                                w_sum=0.0
                                d={}

                                error0=[]
                                t1=0
                                t2 = 0
                                t3 = 0
                                t4 = 0
                                t5 = 0
                                t6 = 0
                                times1=times1+1
                                t=0

                                while w_sum<=TH:
                                    w_sum = 0.0
                                    t=t+1
                                    # print(t)
                                    a=select_testdata(k,begin_edge,j,test_data)

                                    b=result_RP(begin_edge,a[0][0],a[0][1])
                                    dic={}
                                    for key in d:
                                        if d.get(key):  # 判断dictb中是否存在这个key
                                            dic[key] = d[key] + b[0][key]
                                        else:
                                            dic[key] = d[key]
                                    for key in b[0]:
                                        if d.get(key):
                                            pass
                                        else:
                                            dic[key] = b[0][key]
                                    d=sorted(dic.items(),key = lambda x:x[1],reverse = True)
                                    x1=[]
                                    y1=[]
                                    for i in range(0, len(d)):
                                        x1.append(d[i][0])
                                        y1.append(d[i][1])
                                    d = dict(zip(x1, y1))
                                    d1=b[0]


                                    result1=list(d.items())[0:knn]
                                    result = list(d1.items())[0:knn]
                                    # print('定位结果RP')
                                    # print(k,result1)
                                    test_xy=RP_xy[k]

                                    error0.append(Calculate_coordinates(test_xy,RP_xy,result1,knn))
                                    # if error0[-1]<=4  and t1==0:
                                    #     t1=t
                                    # if error0[-1] <= 3.5 and t2 == 0:
                                    #     t2 = t
                                    #
                                    # if error0[-1]<=3  and t3==0:
                                    #     t3=t
                                    # if error0[-1] <= 2.5 and t4 == 0:
                                    #     t4 = t
                                    #
                                    # if error0[-1]<=2 and t5==0:
                                    #     t5=t
                                    # if error0[-1] <= 1.5 and t6 == 0:
                                    #     t6 = t
                                    if random_select==0:
                                        Edge=select_edge.select_edge(result1, begin_Node, begin_edge,knn)[0]

                                        Node=select_edge.select_edge(result1, begin_Node, begin_edge,knn)[1]
                                        print(Edge,Node)
                                    else:

                                        Edge=np.random.choice(Edge_data[begin_Node])
                                        while Edge==begin_edge:
                                            Edge = np.random.choice(Edge_data[begin_Node])
                                        # print(Edge)
                                        Node=np.random.choice(Node_data[Edge])

                                        while Node == begin_Node:
                                            Node = np.random.choice(Node_data[Edge])
                                        if Node ==13:
                                            Node=12
                                        if Node ==14:
                                            Node=11
                                        if Node ==15:
                                            Node=9
                                        # print(Edge, Node)
                                    begin_edge=Edge
                                    begin_Node=Node
                                    # print(begin_edge)
                                    for i in range(0,knn):#xiugai
                                        w_sum=w_sum+result1[i][1]

                                # print(Edge)
                                # print("node",Node)
                                print("误差",error0)
                                # if error0[-1]>5:
                                #     RP_error5.append(k)
                                error.append(error0[-1])#-1
                                times=times+len(error0)
                                len1.append(len(error0))
                                print(error,len(error0))

                df = pd.DataFrame(len1)
                df.to_csv("3edge k=" + str(knn)+ " TH=" + str(TH) + "" + ALGO + "")  # 1to43-18to20
                df= pd.DataFrame(error)
                df.to_csv("3k="+str(knn)+"knn="+str(knn)+" TH="+str(TH)+""+ALGO+""+str(times/times1)+"")#1to43-18to20
                print("走了n条边",times/times1)
                hist, bins = np.histogram(error, bins=30)  #, range=(0, 10))
                pdf = hist / sum(hist)
                cdf = np.cumsum(pdf)
                print(cdf)
                f=interp1d(cdf,bins[1:])
                plt.plot(bins[1:], cdf, label='TH='+str(TH)+','+ALGO+'')
                # plt.plot(bins[1:], pdf, label="PDF")
                y = [0.8, 0.9, 0.95]
                x = []

    #             for i in y:
    #                 a=f(i).tolist()
    #                 x.append(f(i))
    #             print(x)
    #             x_three.append(x)
    #             for i in range(0,len(y)):
    #                 try:
    #                     plt.plot(x[i], y[i], 'r.')
    #                     x[i]=float(x[i].tolist())
    #
    #                     # plt.text(x[i],y[i], (round(x[i],2)), color='r')#, y[i]), color='r')
    #                 except:
    #                     continue
    #         plt.xlim(0,None)
    #         plt.ylim(0,None)
    # print(x_three)
    # print(RP_error5)
    # plt.title("TH=4")#18-20Edge1to43RPTH=4
    # plt.legend()
    # path = './pic/'
    # plt.show()
