"""
合并两个表中的数据，并根据 SA 字段进行分类，最后将分类后的数据分成50段并写入到对应的表中。
同时，删除掉有length长度为0的数据。
"""
import multiprocessing

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from functools import partial

# 数据库配置
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'zdll.123.'
DB_NAME = 'test'

# 配置
FRACTIONAL_NUMBER = 32  # 分成 50 段

number_to_points_0 = {
            1: [(3, 0), (5, -3)],
            2: [(5, -3), (10, -3)],
            3: [(10, -3), (11, 3)],
            4: [(11, 3), (7, 4)],
            5: [(7, 4), (3, 0)]  # 循环到 1 的情况
        }

number_to_points_1 = {
    1: [(10, -3), (7, 4)],
    2: [(7, 4), (10, -3)]
}

number_to_points_2 = {
    1: [(3, 0), (-1, 2)],
    2: [(-1, 2), (-1, -3)],
    3: [(-1, -3), (3, 0)]
}

number_to_points_3 = {
    1: [(-1, 6), (4, 7)],
    2: [(4, 7), (-1, 6)]
}

number_to_points_4 = {
    1: [(-1, 2), (4, 7)],
    2: [(4, 7), (4, 11)],
    3: [(4, 11), (-1, 6)],
    4: [(-1, 6), (-1, 2)]
}

number_to_points_5 = {
    1: [(-1, -3), (-1, -12)],
    2: [(-1, -12), (1, -7)],
    3: [(1, -7), (-1, -3)]
}

number_to_points_6 = {
    1: [(4, 11), (4, 16)],
    2: [(4, 16), (4, 11)]
}

number_to_points_7 = {
    1: [(-1, -12), (-1, -17)],
    2: [(-1, -17), (-1, -12)]
}

number_to_points_8 = {
    1: [(1, -7), (6, -7)],
    2: [(6, -7), (1, -7)]
}


def points_on_segment(point1, point2, distance):
    x1, y1 = point1
    x2, y2 = point2

    # 计算两点之间的距离
    total_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 将方向向量归一化
    dx = (x2 - x1) / total_distance
    dy = (y2 - y1) / total_distance

    # 计算距离点t1和点t2
    point1_new = (x1 + distance * dx, y1 + distance * dy)
    point2_new = (x2 - distance * dx, y2 - distance * dy)

    return np.round(point1_new, 2), np.round(point2_new, 2)


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def project_point(px, py, x1, y1, x2, y2):
    C = x2 - x1
    D = y2 - y1
    len_sq = C * C + D * D
    param = ((px - x1) * C + (py - y1) * D) / len_sq if len_sq != 0 else 0
    param = max(0, min(1, param))
    xx = x1 + param * C
    yy = y1 + param * D
    return xx, yy


# 合并两个表中的数据
def combined_data(args):
    which_times, which_sides = args
    engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
    # 读取数据
    df_tf_position = pd.read_sql(f'SELECT * FROM {which_sides}{which_times}tf_position WHERE number != 0', con=engine, parse_dates=['time'])
    df_rece_rss = pd.read_sql(f'SELECT * FROM {which_sides}{which_times}rece_rss', con=engine, parse_dates=['time'])

    # 保留毫秒部分
    df_tf_position['time'] = df_tf_position['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df_rece_rss['time'] = df_rece_rss['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # 将时间列转换为datetime格式
    df_tf_position['time'] = pd.to_datetime(df_tf_position['time'])
    df_rece_rss['time'] = pd.to_datetime(df_rece_rss['time'])

    # 标记变化点
    df_tf_position['prev_x'] = df_tf_position['position_x'].shift(1)
    df_tf_position['prev_y'] = df_tf_position['position_y'].shift(1)
    df_tf_position['changed'] = ((df_tf_position['position_x'] != df_tf_position['prev_x']) |
                                 (df_tf_position['position_y'] != df_tf_position['prev_y']))
    df_tf_position.loc[0, 'changed'] = False

    # 初始化新的数据表
    merged_data = []

    # 合并数据
    for index, tf_row in df_tf_position.iterrows():
        if tf_row['changed']:
            if index == 0:
                time_start = tf_row['time']
            else:
                time_start = tf_row['time'] - (tf_row['time'] - df_tf_position.loc[index - 1, 'time']) / 2

            if index == len(df_tf_position) - 1:
                time_end = tf_row['time']
            else:
                time_end = tf_row['time'] + (df_tf_position.loc[index + 1, 'time'] - tf_row['time']) / 2

            df_rece_rss_window = df_rece_rss[(df_rece_rss['time'] >= time_start) & (df_rece_rss['time'] < time_end)]

            if not df_rece_rss_window.empty:
                for rss_index, rss_row in df_rece_rss_window.iterrows():
                    merged_row = {'id': tf_row['id'],
                                  'rss': rss_row['rss'],
                                  'time_tf': tf_row['time'],
                                  'time_rss': rss_row['time'],
                                  'position_x': tf_row['position_x'],
                                  'position_y': tf_row['position_y'],
                                  'SA': rss_row['SA'],
                                  'DA': rss_row['DA'],
                                  'times': tf_row['times'],
                                  'number': tf_row['number']}
                    merged_data.append(merged_row)
            # 如果没有数据，则给每个 SA 添加一条数据，rss 为 -95，目的是防止tf_position中的数据丢失
            else:
                for SA in ['f4:e7', 'cf:57', 'ec:1d', 'e4:df', 'ba:02', '81:45']:
                    merged_row = {'id': tf_row['id'],
                              'rss': -95,
                              'time_tf': tf_row['time'],
                              'time_rss': tf_row['time'],
                              'position_x': tf_row['position_x'],
                              'position_y': tf_row['position_y'],
                              'SA': f'00:00:00:00:{SA}',
                              'DA': '00:00:00:00:00:00',
                              'times': tf_row['times'],
                              'number': tf_row['number']}
                merged_data.append(merged_row)

    # 转换为DataFrame
    df_merged = pd.DataFrame(merged_data)

    df_merged = df_merged[(df_merged['times'] != 0) & (df_merged['number'] != 0)]

    # 保留毫秒部分格式
    df_merged['time_tf'] = df_merged['time_tf'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df_merged['time_rss'] = df_merged['time_rss'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # 将合并后的数据写入数据库
    df_merged.to_sql(f'{which_sides}{which_times}_merged_table', con=engine, if_exists='replace', index=False)

    print('数据合并完成并写入到数据库。')

    # 数据分类
    categories = {'f4e7': 'f4:e7', 'cf57': 'cf:57', 'ec1d': 'ec:1d', 'e4df': 'e4:df', 'ba02': 'ba:02', '8145': '81:45'}
    for category, suffix in categories.items():
        # 根据 SA 字段进行分类
        df_category = df_merged[df_merged['SA'].str[-5:] == suffix]

        # 将分类后的数据写入对应的表
        df_category.to_sql(f'{which_sides}{which_times}_{category}_table', con=engine, if_exists='replace', index=False)

    print('数据分类完成并写入到对应的表。')

    classified_data(which_sides, which_times, engine)


def classified_data(which_sides, which_times, engine):
    # 读取分类后的表
    i = 1
    categories = {'f4e7': 1, 'cf57': 2, 'ec1d': 3, 'e4df': 4, 'ba02': 5, '8145': 6}
    sa_list = ['f4:e7', 'cf:57', 'ec:1d', 'e4:df', 'ba:02', '81:45']
    for category, rp_num in categories.items():
        df = pd.read_sql(f'SELECT * FROM {which_sides}{which_times}_{category}_table', con=engine)

        # 删除 number=0 的数据
        df = df[df['number'] != 0]

        # 转换 position_x 、 position_y 和 times列为数值类型
        df['position_x'] = pd.to_numeric(df['position_x'], errors='coerce')
        df['position_y'] = pd.to_numeric(df['position_y'], errors='coerce')
        df['times'] = pd.to_numeric(df['times'], errors='coerce')
        df['rss'] = pd.to_numeric(df['rss'], errors='coerce')

        # 初始化新的数据表
        new_data = []

        # 设置每个 number 对应的 start_point 和 end_point

        switcher = {
            '': number_to_points_0,
            '1': number_to_points_1,
            '2': number_to_points_2,
            '3': number_to_points_3,
            '4': number_to_points_4,
            '5': number_to_points_5
        }
        number_to_points = switcher[which_sides]
        need_to_remove = []
        # 遍历每个 times 的变化点
        for times in df['times'].unique():
            if times == max(df['times']):
                break
            df_times = df[df['times'] == times].sort_values('time_tf')

            for number in df_times['number'].unique():
                df_number = df_times[df_times['number'] == number].sort_values('time_tf')
                df_number_rss_mean = df_number['rss'].mean()

                # 获取当前 number 对应的 start_point 和 end_point
                start_point, end_point = number_to_points[int(number)]
                start_point, end_point = points_on_segment(start_point, end_point, 0.2)

                # 创建 31 个分段点（包括起点和终点）
                segments = []
                for i in range(1, FRACTIONAL_NUMBER):
                    point = (np.round(start_point[0] + (end_point[0] - start_point[0]) * (i / FRACTIONAL_NUMBER), 2),
                             np.round(start_point[1] + (end_point[1] - start_point[1]) * (i / FRACTIONAL_NUMBER), 2))
                    segments.append(point)

                # 计算每段的时间范围
                segment_times = []
                for segment_point in segments:
                    projected_points = df_number.apply(
                        lambda row: project_point(row['position_x'], row['position_y'], start_point[0], start_point[1], end_point[0], end_point[1]),
                        axis=1
                    )
                    distances = projected_points.apply(
                        lambda p: calculate_distance(p, segment_point)
                    )
                    closest_index = distances.idxmin()
                    closest_time = df_number.loc[closest_index, 'time_tf']
                    segment_times.append(closest_time)

                for length, segment_time in enumerate(segment_times, start=1):
                    if length == len(segment_times):
                        break
                    df_segment = df[(segment_time <= df['time_tf']) & (df['time_tf'] < segment_times[length]) & (
                                df['number'] == number) & (df['times'] == times)].copy()

                    # df_number_rss_mean 大于-80的情况，如果长度小于3，则删除
                    if len(df_segment) < 3 and df_number_rss_mean > -80:
                        need_to_remove.append([str(times), str(number)])
                        break
                    # 如果没有数据，则给当前 SA 添加一条数据，rss 为 -95
                    if len(df_segment) == 0:
                        new_rows = []
                        merged_row = {'id': df_number['id'].values[0], 'rss': -95, 'time_tf': segment_time,
                                      'time_rss': segment_time, 'position_x': segments[length - 1][0],
                                      'position_y': segments[length - 1][1], 'SA': f'50:fa:84:cb:{sa_list[rp_num - 1]}',
                                      'DA': '00:00:00:00:00:00', 'times': times, 'number': number}
                        new_rows.append(merged_row)
                        if i:
                            print(category, new_rows)
                            i = 0

                        df_segment = pd.concat([df_segment, pd.DataFrame(new_rows)], ignore_index=True)
                    df_segment['length'] = length
                    new_data.append(df_segment)

        # 合并所有新数据
        if new_data:
            new_df = pd.concat(new_data)

            # 确保 length 按照从 1 到 25 排序
            new_df = new_df.sort_values(by=['times', 'length'])

            # 统计每个 times 和 number 组合下不同 length 值的数据条目数
            # length_counts = new_df.groupby(['times', 'number', 'length']).size().reset_index(name='count')

            # 根据需要删除的组合过滤数据
            new_df['times'] = new_df['times'].astype(str)
            new_df['number'] = new_df['number'].astype(str)
            new_df = new_df[~new_df.set_index(['times', 'number']).index.isin(need_to_remove)]
######################################################################################################################
            # 将处理后的数据按 number 分类保存到新的表
            for number in new_df['number'].unique():
                num = int(number)
                num1 = rp_num + int(which_times) * 6  # 这里将 num1 设为 RP 的编号
                new_table_name = f'side_6_rp_{num1}_{num}'
                df_number = new_df[new_df['number'] == number]
                df_number.to_sql(new_table_name, con=engine, if_exists='replace', index=False)
            print('数据处理完成并写入到新的表中。')
        else:
            print('没有符合条件的数据需要处理。')


if __name__ == "__main__":
    Which_sides = '1'
    Which_timess = ['0', '1', '2', '3', '4', '5']

    pool = multiprocessing.Pool(len(Which_timess))

    # 使用进程池并行处理数据
    args = [(which_times, Which_sides) for which_times in Which_timess]
    pool.map(combined_data, args)

    # 关闭进程池
    pool.close()
    pool.join()
