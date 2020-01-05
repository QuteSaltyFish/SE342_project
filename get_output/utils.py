import numpy as np
import cv2
import math


def error(p, x, y):
    return np.sum(np.square(p[0]*x + p[1] - y))


def img_label(origin, result):
    # origin: 原图
    # result: unet输出 + 处理后的结果
    # 两者shape应该相等

    dst = origin.copy()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, _, stats, centroid = cv2.connectedComponentsWithStats(
        gray, connectivity=8)  # 提取连通分量的边框和中心值

    # 去除背景
    stats = stats[1:, ...]
    centroid = centroid[1:, ...].astype(np.int32)

    # 给原图中的瓶盖染色
    indices = np.sum((result != (0, 0, 0)), axis=-1, dtype=np.bool)
    dst[indices, :] = result[indices, :]

    for i in range(len(stats)):
        cx, cy = centroid[i]
        x, y, w, h, _ = stats[i]
        black = (0, 0, 0)
        string = '({}, {})'.format(cx, cy)

        if result[cy, cx, 1] == 255:  # 侧着
            conn_offset = np.where(result[y:y+h, x:x+w, 1] == 255)  # 连通分量

            p1 = np.polyfit(conn_offset[1], conn_offset[0], 1)
            p2 = np.polyfit(conn_offset[0], conn_offset[1], 1)

            # 选择误差较小的拟合
            if error(p1, conn_offset[1], conn_offset[0]) < error(p2, conn_offset[0], conn_offset[1]):
                angle = int(math.atan2(1, p1[0]) / math.pi * 180)
            else:
                angle = int(math.atan2(p2[0], 1) / math.pi * 180)

                if angle < 0:
                    angle += 180

            string += ' {}deg'.format(angle)

        dst = cv2.putText(dst, string, (x, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 4, black, thickness=3)

    return dst
