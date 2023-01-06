import csv
import matplotlib.pyplot as plt


def saveCsv(path_t, path_x, path_y, path_v, path_yaw, path_a, path_steer, path_steer_rate, init_x, init_y, sampleT,
            exp_name, path_num):
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






