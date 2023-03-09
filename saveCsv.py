import csv
import matplotlib.pyplot as plt
import numpy as np


def saveCsv(path_t, path_x, path_y, path_v, path_yaw, path_a, path_steer, path_steer_rate, init_x, init_y, sampleT,
            exp_name, path_num, args=None, data_num=0):
    with open('./Result/case-{}/{}-result-{}.csv'.format(path_num, exp_name, path_num), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_t)):
            writer.writerow([path_t[i], path_x[i], path_y[i], path_yaw[i]])
    with open('./Result/case-{}/{}-result-state-{}.csv'.format(path_num, exp_name, path_num), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_t)):
            writer.writerow([path_t[i], path_x[i], path_y[i], path_v[i], path_yaw[i], path_steer[i]])
    with open('./Result/case-{}/{}-result-control-{}.csv'.format(path_num, exp_name, path_num), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_a)):
            writer.writerow([path_t[i], path_a[i], path_steer_rate[i]])
    fig, ax = plt.subplots()
    ax.plot(path_x, path_y, 'go', ms=3, label='optimized path')
    ax.plot(init_x, init_y, 'ro', ms=3, label='init path')
    ax.legend()
    plt.savefig("./Result/case-{}/{}-err-traj-{}.svg".format(path_num, exp_name, path_num))

    fig2, ax2 = plt.subplots(4)
    plt.subplots_adjust(hspace=0.35)
    t_v = [sampleT*k for k in range(len(path_v))]
    t_a = [sampleT*k for k in range(len(path_a))]
    t_steer = [sampleT * k for k in range(len(path_steer))]
    t_steer_rate = [sampleT*k for k in range(len(path_steer_rate))]
    ax2[0].plot(t_v, path_v, label='v-t')
    ax2[1].plot(t_a, path_a, label='a-t')
    ax2[2].plot(t_steer, path_steer, label='steer-t')
    ax2[3].plot(t_steer_rate, path_steer_rate, label='steer-rate-t')
    ax2[0].legend()
    ax2[1].legend()
    ax2[2].legend()
    ax2[3].legend()
    plt.savefig("./Result/case-{}/{}-kina-{}.svg".format(path_num, exp_name, path_num))

    if args.gen_npy:
        # save numpy array for training
        outfile_x = "./Result/case-{}/data_{}/array_x".format(path_num, data_num)
        outfile_y = "./Result/case-{}/data_{}/array_y".format(path_num, data_num)
        outfile_v = "./Result/case-{}/data_{}/array_v".format(path_num, data_num)
        outfile_yaw = "./Result/case-{}/data_{}/array_yaw".format(path_num, data_num)
        outfile_a = "./Result/case-{}/data_{}/array_a".format(path_num, data_num)
        outfile_steer = "./Result/case-{}/data_{}/array_steer".format(path_num, data_num)
        outfile_steer_rate = "./Result/case-{}/data_{}/array_steer_rate".format(path_num, data_num)
        outfile_t = "./Result/case-{}/data_{}/array_t".format(path_num, data_num)

        np.save(outfile_t, path_t)
        np.save(outfile_x, path_x)
        np.save(outfile_y, path_y)
        np.save(outfile_v, path_v)
        np.save(outfile_yaw, path_yaw)
        np.save(outfile_a, path_a)
        np.save(outfile_steer, path_steer)
        np.save(outfile_steer_rate, path_steer_rate)





