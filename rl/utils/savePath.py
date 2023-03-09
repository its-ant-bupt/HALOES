import matplotlib.pyplot as plt
import csv
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def show(path, case, path_num, savePtah):
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.xlim(case.xmin, case.xmax)
    plt.ylim(case.ymin, case.ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_axisbelow(True)
    plt.title('Case %d' % (path_num))
    plt.grid(linewidth=0.2)
    plt.xlabel('X / m', fontsize=14)
    plt.ylabel('Y / m', fontsize=14)
    for j in range(0, case.obs_num):
        plt.fill(case.obs[j][:, 0], case.obs[j][:, 1], facecolor='k', alpha=0.5)

    temp = case.vehicle.create_polygon(case.x0, case.y0, case.theta0)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='green')
    temp = case.vehicle.create_polygon(case.xf, case.yf, case.thetaf)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='red')

    for i in range(len(path.x)):
        temp = case.vehicle.create_polygon(path.x[i], path.y[i], path.yaw[i])
        # plot_polygon(temp, fill=False, color='b')
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.15, color='blue')
        # plt.plot(path.x[i], path.y[i], marker='.', color='red', markersize=0.5)
    plt.plot(path.x, path.y, color='red', linewidth=0.1)

    plt.savefig(savePtah)
    # plt.show()


def showZone(path, case, path_num, savePtah):

    # plt.subplot(1, 2, 1)
    fig, ax = plt.subplots(1, 1)
    # ax.set_xlim(case.xmin, case.xmax)
    # ax.set_xlim(case.ymin, case.ymax)
    plt.xlim(case.xmin, case.xmax)
    plt.ylim(case.ymin, case.ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_axisbelow(True)
    plt.title('Case %d' % (path_num))
    plt.grid(linewidth=0.2)
    plt.xlabel('X / m', fontsize=14)
    plt.ylabel('Y / m', fontsize=14)
    for j in range(0, case.obs_num):
        ax.fill(case.obs[j][:, 0], case.obs[j][:, 1], facecolor='k', alpha=0.5)

    temp = case.vehicle.create_polygon(case.x0, case.y0, case.theta0)
    ax.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='green')
    temp = case.vehicle.create_polygon(case.xf, case.yf, case.thetaf)
    ax.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='red')

    for i in range(len(path.x)):
        temp = case.vehicle.create_polygon(path.x[i], path.y[i], path.yaw[i])
        # plot_polygon(temp, fill=False, color='b')
        ax.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.15, color='blue')
        # plt.plot(path.x[i], path.y[i], marker='.', color='red', markersize=0.5)
    ax.plot(path.x, path.y, color='red', linewidth=0.1)

    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.5, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)
    for i in range(len(path.x)):
        temp = case.vehicle.create_polygon(path.x[i], path.y[i], path.yaw[i])
        # plot_polygon(temp, fill=False, color='b')
        axins.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.15, color='blue')
    axins.plot(path.x, path.y, color='red', linewidth=0.1)


    axins.set_xlim(-16.7, -16)
    axins.set_ylim(-2.5, -2)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.savefig(savePtah)
    # plt.show()


def saveCsv(path_t, path_x, path_y, path_v, path_yaw, path_a, path_steer, path_steer_rate, init_x, init_y, sampleT,
            save_path, i, j, case_num):
    with open(os.path.join(save_path, 'csv/case{}-{}-{}-result.csv'.format(case_num, i, j)), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_t)):
            writer.writerow([path_t[i], path_x[i], path_y[i], path_yaw[i]])
    with open(os.path.join(save_path, 'csv/case{}-{}-{}-result-state.csv'.format(case_num, i, j)), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_t)):
            writer.writerow([path_t[i], path_x[i], path_y[i], path_v[i], path_yaw[i], path_steer[i]])
    with open(os.path.join(save_path, 'csv/case{}-{}-{}-result-control.csv'.format(case_num, i, j)), 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        for i in range(len(path_a)):
            writer.writerow([path_t[i], path_a[i], path_steer_rate[i]])
    fig, ax = plt.subplots()
    ax.plot(path_x, path_y, 'go', ms=3, label='optimized path')
    ax.plot(init_x, init_y, 'ro', ms=3, label='init path')
    ax.legend()
    plt.savefig(os.path.join(save_path, 'svg/case{}-{}-{}-err-traj.svg'.format(case_num, i, j)))

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
    plt.savefig(os.path.join(save_path, 'svg/case{}-{}-{}-kina.svg'.format(case_num, i, j)))

