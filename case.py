import numpy as np
import csv
from Vehicle import Vehicle


class Case:
    def __init__(self):
        self.x0, self.y0, self.theta0 = 0, 0, 0
        self.xf, self.yf, self.thetaf = 0, 0, 0
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.obs_num = 0
        self.obs = np.array([])
        self.vehicle = Vehicle()

    @staticmethod
    def read(file):
        case = Case()
        with open(file, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]
            case.x0, case.y0, case.theta0 = v[0:3]
            case.xf, case.yf, case.thetaf = v[3:6]
            case.xmin = min(case.x0, case.xf) - 10
            case.xmax = max(case.x0, case.xf) + 10
            case.ymin = min(case.y0, case.yf) - 10
            case.ymax = max(case.y0, case.yf) + 10
            # case.xmin = min(case.x0, case.xf)
            # case.xmax = max(case.x0, case.xf)
            # case.ymin = min(case.y0, case.yf)
            # case.ymax = max(case.y0, case.yf)

            case.obs_num = int(v[6])  # 获取障碍物数目
            num_vertexes = np.array(v[7:7 + case.obs_num], dtype=np.int)  # 获取每个障碍物的边数
            case.num_vertexes = num_vertexes
            case.obsLineList = v[7 + case.obs_num:]
            # 计算每个障碍物顶点坐标的开始位置
            vertex_start = 7 + case.obs_num + (np.cumsum(num_vertexes, dtype=np.int) - num_vertexes) * 2
            case.obs = []
            for vs, nv in zip(vertex_start, num_vertexes):
                # 添加每个障碍物顶点的坐标
                case.obs.append(np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
        return case