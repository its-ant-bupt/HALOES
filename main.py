#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path

import numpy as np
import time
import csv
from case import Case
from Vehicle import Vehicle, Path, OBCAPath
from Show import show
from quadraticOBCA import quadraticPath
from pyobca.search import *
from pympc.mpc_optimizer import MPCOptimizer
from saveCsv import saveCsv
import argparse


def get_argparse():
    parse = argparse.ArgumentParser(description="Information of Automated Parking.")
    parse.add_argument("--path_num", type=int, default=3, help="The number of case.")
    parse.add_argument("--exp_name", type=str, default='test', help="The exp name to save.")
    parse.add_argument("--trans", action="store_true", default=False, help="Whether to trans the point to (0,0,0).")
    parse.add_argument("--sample_time", type=float, default=0.15, help="The sample time.")
    parse.add_argument("--pre_length", type=int, default=5, help="The prediction horizon.")
    parse.add_argument("--obca_sample_time", type=float, default=0.1, help="The sample time of OBCA.")
    parse.add_argument("--obca_gap", type=int, default=1, help="The gap of obca.")
    parse.add_argument("--gen_npy", action="store_true", default=False, help="Generate the numpy data of trajectory.")
    parse.add_argument("--data_num", type=int, default=3, help="The number of generate numpy data.")
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    path_num = args.path_num
    exp_name = args.exp_name
    use_trans = args.trans
    case = Case.read('BenchmarkCases/Case%d.csv' % path_num)
    vehicle = Vehicle()
    # 模型预测轨迹规划
    mpc_optimizer = MPCOptimizer(case, vehicle, args)
    mpc_optimizer.initialize()
    mpc_optimizer.build_model()
    mpc_optimizer.generate_object(disCostFinal=50000, deltaCostFinal=10000, disCost=1000, deltaCost=5000,
                                  aCost=0, steerCost=0, obsPower=1.6)
    mpc_optimizer.generate_constrain()
    final_path, initQuadraticPath = mpc_optimizer.solve()
    # OBCA二次优化
    cfg = VehicleConfig()
    cfg.T = args.sample_time
    gap = args.obca_gap
    sampleT = args.obca_sample_time
    obstacles = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obstacles.append(obs)
    path_x, path_y, path_v, path_yaw, path_steer, path_a, path_steer_rate = quadraticPath(
                                            initialQuadraticPath=initQuadraticPath, obstacles=obstacles,
                                            vehicle=vehicle, max_x=case.xmax, max_y=case.ymax,
                                            min_x=case.xmin, min_y=case.ymin,
                                            gap=gap, cfg=cfg, sampleT=sampleT)
    # 画图
    obcaPath = Path(path_x, path_y, path_yaw)
    if not os.path.exists("./Result/case-{}".format(path_num)):
        os.mkdir("./Result/case-{}".format(path_num))
    if args.gen_npy and not os.path.exists("./Result/case-{}/data_{}".format(path_num, args.data_num)):
        os.mkdir("./Result/case-{}/data_{}".format(path_num, args.data_num))
    show(final_path, case, path_num, exp_name+"-init", args, data_num=args.data_num)
    show(obcaPath, case, path_num, exp_name+"-obca", args, data_num=args.data_num)
    obcaPath_5gap = Path(path_x[::5], path_y[::5], path_yaw[::5])
    show(obcaPath_5gap, case, path_num, exp_name+"-5gap", args, data_num=args.data_num)
    path_t = [sampleT * k for k in range(len(path_x))]
    saveCsv(path_t=path_t, path_x=path_x, path_y=path_y, path_v=path_v, path_yaw=path_yaw, path_a=path_a,
            path_steer=path_steer, path_steer_rate=path_steer_rate, init_x=final_path.x, init_y=final_path.y,
            sampleT=sampleT, exp_name=exp_name, path_num=path_num, args=args, data_num=args.data_num)



