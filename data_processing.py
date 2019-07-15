import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from sklearn import decomposition
from sklearn.manifold import TSNE
import json


def read_csv(filename):
    d = pd.read_csv(filename, header=None, sep='\t')
    d.columns = ['t', 'i', 'j', 'Ci', 'Cj']
    return d


def read_txt(filename):
    d = pd.read_csv(filename, header=None, sep='\t')
    d.columns = ['i', 'Ci', 'Gi']
    return d


# read contact list and output data in single time windows
def single_time_window():
    # 原始数据（已添加列标签）
    raw_data = read_csv("highschool_2012.csv")

    # 计算最早的contact时间
    start_time = raw_data.min(axis=0)['t']

    # 将每个contact的时间替换成距离最早contact时间的差值
    for i in range(0, len(raw_data)):
        raw_data.at[i, 't'] = raw_data.iloc[i]['t'] - start_time

    # 生成所有时间窗的范围
    interval = 60 * 60  # 60 min 时间窗长度
    increment = 6 * 60  # 6 min 时间窗每次移动的增量
    s = 0  # 时间窗开始时间
    e = s + interval  # 时间窗结束时间
    range_array = np.array([s, e])

    # 在时间窗未超过最迟的contact时间内循环
    while e < raw_data.iloc[len(raw_data)-1]['t']:
        s += increment
        e = s + interval
        tmp = np.array([s, e])
        # 写入时间窗数组
        range_array = np.vstack((range_array, tmp))

    matrix_list = []
    for j in range(0, len(range_array)):
        print("current window index:" + str(j))
        # 从原始数据中获取对应时间窗内的数据                 #第j个时间窗                           #第j个
        selected = raw_data[(raw_data['t'] >= range_array[j][0]) & (raw_data['t'] <= range_array[j][1])]
        selected = selected.reindex()  # 重新计算索引

        adj_matrix = np.zeros([180, 180])

        # 读取meta_data
        meta_data = read_txt("metadata_2012.txt")
        # 在所有选出的数据(即可以看做图的边数据)中将i，j(即每次contact的2人的匿名ID)在meta_data中查找替换为顺序的索引
        # 即如600替换为0， 601替换为1
        # 并在邻接矩阵中对应单元边权重+1
        for i in range(0, len(selected)):
            rep_i = meta_data[meta_data['i'] == selected.iloc[i]['i']].index[0]  # 替换后的i
            rep_j = meta_data[meta_data['i'] == selected.iloc[i]['j']].index[0]  # 替换后的j
            adj_matrix[rep_i][rep_j] += 1

        # adj = pd.DataFrame(adj_matrix)
        # w = []  # 边权重数据 用于python中debug时绘制出边的width，对于导出json数据没有用处
        #
        # G = nx.Graph()
        # for i in range(0, 180):
        #     for j in range(0, 180):
        #         # G.add_node(i)  # 如果将未有contact节点也显示，则取消注释
        #         if adj.iloc[i][j] > 0.0:
        #             G.add_edge(i, j, weight=adj.iloc[i][j])
        #             w.append(adj.iloc[i][j] / 5)
        # # Draw Graph to Debug
        # positions = nx.spring_layout(G)
        #
        # nx.draw_networkx_nodes(G, pos=positions, hold=True, with_labels=False, node_size=10)
        # nx.draw_networkx_edges(G, pos=positions, width=np.array(w))
        # plt.show()

        matrix_list.append(adj_matrix)

    return matrix_list


def dimension_reduction(matrix):
    vector = []
    for index in range(0, len(matrix)):
        v = matrix[index].reshape([1, 32400]).tolist()[0]
        vector.append(v)

    # pca = decomposition.PCA(n_components=2)
    # pca.fit(vector)
    # coords = pca.transform(vector)
    coords = TSNE(n_components=2).fit_transform(vector)
    np.save("all_windows.npy", np.array(coords))


def debug_graph(g, w):
    # Draw Graph to Debug
    positions = nx.spring_layout(g)

    nx.draw_networkx_nodes(g, pos=positions, hold=True, with_labels=False, node_size=10)
    nx.draw_networkx_edges(g, pos=positions, width=np.array(w))
    plt.show()


def output_json(data, filename):
    # Write data into json file
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # m_list = np.array(single_time_window())
    # np.save("matrix_list.npy", m_list)
    matrix_array = np.load("matrix_list.npy")

    # dimension_reduction(matrix_array)
    coords = np.load("all_windows.npy")
    # plt.plot(coords[:, 0], coords[:, 1], linewidth=0.5)
    # plt.scatter(coords[:, 0], coords[:, 1], marker='o', s=5, c='r')
    # plt.show()

    dict_data = []
    for k in range(0, len(matrix_array)):
        print("current index:" + str(k))
        adj_m = matrix_array[k]
        adj = pd.DataFrame(adj_m)
        w = []  # 边权重数据 用于python中debug时绘制出边的width，对于导出json数据没有用处

        G = nx.Graph()
        for i in range(0, 180):
            for j in range(0, 180):
                # G.add_node(i)  # 如果将未有contact节点也显示，则取消注释
                if adj.iloc[i][j] > 0.0:
                    G.add_edge(i, j, weight=adj.iloc[i][j])
                    # w.append(adj.iloc[i][j] / 5)

        data = json.loads('{"vector":[' + str(coords[k, 0]) + ',' + str(coords[k, 1]) + '],"graph":' + json.dumps(json_graph.node_link_data(G)) + '}')
        dict_data.append(data)

    data = json.dumps(dict_data)
    output_json(json.loads(data), "fake_vector_data.json")
