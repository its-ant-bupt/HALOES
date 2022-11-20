import pyobca
import numpy as np
import math as m


def quadraticPath(initialQuadraticPath, obstacles, vehicle, max_x, max_y, min_x, min_y, gap=1, cfg=None, sampleT=0.1):
    ds_path = downsample_smooth(initialQuadraticPath, gap, vehicle, sampleT)
    if len(ds_path)<2:
        print('no enough path point')
        return
    init_x = []
    init_y = []
    for state in ds_path:
        init_x += [state.x]
        init_y += [state.y]
    # obca optimization
    optimizer = pyobca.OBCAOptimizer(cfg=cfg)
    optimizer.initialize(ds_path, obstacles, max_x=max_x, max_y=max_y, min_x=min_x, min_y=min_y)
    optimizer.build_model()
    optimizer.generate_constrain()
    optimizer.generate_variable()
    r = [[0.1, 0], [0, 0.1]]
    q = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0., 0],
         [0, 0, 0, 0, 0],
         ]
    optimizer.generate_object(r, q)
    optimizer.solve()

    x_opt = optimizer.x_opt.elements()
    y_opt = optimizer.y_opt.elements()
    v_opt = optimizer.v_opt.elements()
    heading_opt = optimizer.theta_opt.elements()
    steer_opt = optimizer.steer_opt.elements()
    a_opt = optimizer.a_opt.elements()
    steer_rate_opt = optimizer.steerate_opt.elements()


    return x_opt, y_opt, v_opt, heading_opt, steer_opt, a_opt, steer_rate_opt



def downsample_smooth(path, gap, cfg, T=0.1):
    if not path:
        print('no path ')
        return []
    ds_path = path[::gap]
    if len(ds_path) < 3:
        return ds_path
    else:
        for i in range(1, len(ds_path)-1):
            v_1 = (ds_path[i].x-ds_path[i-1].x)/T*m.cos(ds_path[i-1].heading) + \
                (ds_path[i].y-ds_path[i-1].y)/T*m.sin(ds_path[i-1].heading)
            v_2 = (ds_path[i+1].x-ds_path[i].x)/T*m.cos(ds_path[i].heading) + \
                (ds_path[i+1].y-ds_path[i].y)/T*m.sin(ds_path[i].heading)
            v = (v_1 + v_2)/2
            ds_path[i].v = v
        for i in range(len(ds_path)-1):
            ds_path[i].a += (ds_path[i+1].v - ds_path[i].v)/T
            diff_theta = ds_path[i+1].heading-ds_path[i].heading
            direction = 1
            if ds_path[i].v < 0:
                direction = -1
            move_distance = m.hypot((ds_path[i+1].x - ds_path[i].x), (ds_path[i+1].y - ds_path[i].y))
            steer = np.clip(m.atan(diff_theta*cfg.lw/move_distance*direction),
                            -cfg.MAX_STEER, cfg.MAX_STEER)
            ds_path[i].steer = steer
        ds_path[-1] = path[-1]
        return ds_path