import matplotlib.pyplot as plt
from pypoman import plot_polygon


def show(path, case, path_num, exp_name, args=None, data_num=0):
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

    plt.savefig("./Result/case-{}/{}-traj{}.svg".format(path_num, exp_name, path_num))
    if args.gen_npy:
        plt.savefig("./Result/case-{}/data_{}/{}-traj{}.svg".format(path_num, data_num, exp_name, path_num))
    plt.show()

