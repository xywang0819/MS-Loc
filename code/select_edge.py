import numpy as np
import pymysql


def select_edge(weight,begin_Node,begin_edge,knn):
    # Node=[(1),(1,2),(2,3),(3,4),(4,5),(1,5),(3,5),(1,6),(6,7),(1,7),(6,8),(8,9),(9,10),(6,10),(8,10),(7,11),(11,12),(7,12)]#索引为边序号
    # Edge=[(0),(1,5,7,9),(1,2),(2,3,6),(3,4),(4,5,6),(7,8,10,13),(8,9,15,17),(10,11),(11,12),(12,13),(15,16),(16,17)]#索引为点序号
    Node = [(1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (3, 5), (1, 6), (6, 7), (1, 7), (6, 8), (8, 9), (9, 10),
                 (6, 10), (8, 10), (7, 11), (11, 12), (7, 12), (12, 13), (9, 15), (11, 14)]  # 索引为边序号
    Edge = [(0), (1, 5, 7, 9), (1, 2), (2, 3, 6), (3, 4), (4, 5, 6), (7, 8, 10, 13), (8, 9, 15, 17), (10, 11,14),
                 (11, 12, 19),
                 (12, 13,14), (15, 16, 20), (16, 17, 18), (16, 17, 18), (15, 16, 20), (11, 12, 19)]  # 索引为点序号

    a=begin_Node
    b=begin_edge
    edge=[b]

    index=[]
    Edge_weight = []


    for i in range(0,len(Edge[a])):

        edge_next=Edge[a][i]#取可能的下一个边
        index.append(edge_next)
        if edge_next==b:
            Edge_weight.append(0)
            continue
        # RP1=weight[0][0]#前三个权重对应的RP
        # RP2=weight[1][0]
        # RP3=weight[2][0]
        Sum=0

        for q in range(0,knn):#前三个权重
            RP=str(weight[q][0])
            m=str(edge_next)
            RP_weight=weight[q][1]
            ap_position='length_rss' + RP + '_' + m + ''
            conn = pymysql.connect(host='localhost', user='root', password='123456', db='test', port=3306,
                                   charset='utf8')
            cur = conn.cursor()
            cur.execute(
                'select r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25 from 00ap_position '
                'where ap_position= %s', ap_position)
            res = cur.fetchone()
            # print(res)
            conn.close()
            vary=0
            for r in range(0, 25):
                vary=vary+abs(float(res[r]))

            Sum=Sum+vary*RP_weight
        Edge_weight.append(Sum)

    # print(Edge_weight)
    Edge_weight1=dict(zip(index,Edge_weight))
    # print("每条边权重",Edge_weight1)
    # b=Edge[a][Edge_weight.index(max(Edge_weight))]
    b=max(Edge_weight1,key=lambda x:Edge_weight1[x])
    # print("选边",b)
    # b=Edge[a][int(b)]


    edge.append(b)
    Node1 = np.random.choice(Node[b])

    while Node1 == begin_Node:
        Node1 = np.random.choice(Node[b])
    if Node1 == 13:
        Node1 = 12
    if Node1 == 14:
        Node1 = 11
    if Node1 == 15:
        Node1 = 9
    # for i in range(0,len(Node[b])):
    #     if Node[b][i]==begin_Node:
    #         continue
    #     else:
    #         a=Node[b][i]
    a=Node1
    return edge[-1],a
# if __name__ == '__main__':
#     weight=[(1, 0.8372444938790954), (4, 0.04402241472007464), (5, 0.02487939861286463)]
#     begin_Node=1
#     begin_edge=0
#     print(select_edge(weight,begin_Node,begin_edge))