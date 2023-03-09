
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import json
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)

# ================= fig 1===========================
with open('PlotFile/DDPG_case7/avg_score.json') as load_f:
    DDPG_load_dict = json.load(load_f)
with open('PlotFile/DDPG_case7_rel/avg_score.json') as load_f:
    DDPG_Rel_load_dict = json.load(load_f)
with open("PlotFile/PPO_case7/avg_score.json") as load_f:
    PPO_load_dict = json.load(load_f)
with open("PlotFile/SAC_case7/avg_score.json") as load_f:
    SAC_Rel_load_dict = json.load(load_f)
with open("PlotFile/SAC_case7_norel/avg_score.json") as load_f:
    SAC_load_dict = json.load(load_f)
ddpg_data_step = np.array(DDPG_load_dict)[:, 1]
ddpg_data_score = np.array(DDPG_load_dict)[:, 2]
ddpg_rel_data_step = np.array(DDPG_Rel_load_dict)[:, 1]
ddpg_rel_data_score = np.array(DDPG_Rel_load_dict)[:, 2]
ppo_data_step = np.array(PPO_load_dict)[:, 1]
ppo_data_score = np.array(PPO_load_dict)[:, 2]
sac_data_step = np.array(SAC_load_dict)[:, 1]
sac_data_score = np.array(SAC_load_dict)[:, 2]
sac_rel_data_step = np.array(SAC_Rel_load_dict)[:, 1]
sac_rel_data_score = np.array(SAC_Rel_load_dict)[:, 2]


# ================= fig 2===========================
# with open('PlotFile/DDPG_random/test_path_1_score.json') as load_f:
#     DDPG_path_1 = json.load(load_f)
# with open('PlotFile/DDPG_random/test_path_2_score.json') as load_f:
#     DDPG_path_2 = json.load(load_f)
# with open('PlotFile/DDPG_random/test_path_3_score.json') as load_f:
#     DDPG_path_3 = json.load(load_f)
# with open('PlotFile/DDPG_random/test_path_7_score.json') as load_f:
#     DDPG_path_7 = json.load(load_f)
# with open('PlotFile/DDPG_random/test_path_8_score.json') as load_f:
#     DDPG_path_8 = json.load(load_f)
#
# ddpg_path_1_step = np.array(DDPG_path_1)[:, 1][:39]
# ddpg_path_1_score = np.array(DDPG_path_1)[:, 2][:39]
# ddpg_path_2_step = np.array(DDPG_path_2)[:, 1][:39]
# ddpg_path_2_score = np.array(DDPG_path_2)[:, 2][:39]
# ddpg_path_3_step = np.array(DDPG_path_3)[:, 1][:39]
# ddpg_path_3_score = np.array(DDPG_path_3)[:, 2][:39]
# ddpg_path_7_step = np.array(DDPG_path_7)[:, 1][:39]
# ddpg_path_7_score = np.array(DDPG_path_7)[:, 2][:39]
# ddpg_path_8_step = np.array(DDPG_path_8)[:, 1][:39]
# ddpg_path_8_score = np.array(DDPG_path_8)[:, 2][:39]
#
# with open('PlotFile/DDPG_FED_random/Global_path_1_score.json') as load_f:
#     DDPF_Fed_path_1 = json.load(load_f)
# with open('PlotFile/DDPG_FED_random/Global_path_2_score.json') as load_f:
#     DDPF_Fed_path_2 = json.load(load_f)
# with open('PlotFile/DDPG_FED_random/Global_path_3_score.json') as load_f:
#     DDPF_Fed_path_3 = json.load(load_f)
# with open('PlotFile/DDPG_FED_random/Global_path_7_score.json') as load_f:
#     DDPF_Fed_path_7 = json.load(load_f)
# # with open('PlotFile/DDPG_FED_random/Global_path_7_avg_score.json') as load_f:
# #     DDPF_Fed_path_7 = json.load(load_f)
# with open('PlotFile/DDPG_FED_random/Global_path_8_score.json') as load_f:
#     DDPF_Fed_path_8 = json.load(load_f)
# ddpg_fed_path_1_step = np.array(DDPF_Fed_path_1)[:, 1]
# ddpg_fed_path_1_score = np.array(DDPF_Fed_path_1)[:, 2]
# ddpg_fed_path_2_step = np.array(DDPF_Fed_path_2)[:, 1]
# ddpg_fed_path_2_score = np.array(DDPF_Fed_path_2)[:, 2]
# ddpg_fed_path_3_step = np.array(DDPF_Fed_path_3)[:, 1]
# ddpg_fed_path_3_score = np.array(DDPF_Fed_path_3)[:, 2]
# ddpg_fed_path_7_step = np.array(DDPF_Fed_path_7)[:, 1]
# ddpg_fed_path_7_score = np.array(DDPF_Fed_path_7)[:, 2]
# ddpg_fed_path_8_step = np.array(DDPF_Fed_path_8)[:, 1]
# ddpg_fed_path_8_score = np.array(DDPF_Fed_path_8)[:, 2]
#
# datas_1 = [[ddpg_path_1_step, ddpg_path_1_score], [ddpg_fed_path_1_step, ddpg_fed_path_1_score]]
# datas_2 = [[ddpg_path_2_step, ddpg_path_2_score], [ddpg_fed_path_2_step, ddpg_fed_path_2_score]]
# datas_3 = [[ddpg_path_3_step, ddpg_path_3_score], [ddpg_fed_path_3_step, ddpg_fed_path_3_score]]
# datas_4 = [[ddpg_path_7_step, ddpg_path_7_score], [ddpg_fed_path_7_step, ddpg_fed_path_7_score]]
# datas_5 = [[ddpg_path_8_step, ddpg_path_8_score], [ddpg_fed_path_8_step, ddpg_fed_path_8_score]]
# datas_total = [datas_1, datas_2, datas_3, datas_4, datas_5]
# title = ["(a) case 1", "(b) case 2", "(c) case 3", "(d) case 4", "(e) case 5"]

col = [(0.5430834294502115, 0.733917723952326, 0.8593156478277586),  # p=0
       (0.5496501345636293, 0.8032449058054594, 0.5650442137639369),  # p=0.01
       (0.9700115340253749, 0.6256670511341791, 0.367120338331411),  # p=0.1
       (0.8225451749327182, 0.48813533256439834, 0.7114801999231065),  # p=0.02
       (0.17748558246828144, 0.28525951557093426, 0.34726643598615914),  # p=0.2
       (0.5172062027425349, 0.4829424580289632, 0.7259002947584262),  # p=0.05
       (0.845925925925926, 0.2324080481865949, 0.1541817249775727),  # p=0.5
       (0.7174112520825324, 0.14635909265667052, 0.5073176983211585),  # p=0.9
       "b"]



# keys = load_dict.keys()
# print(keys)
# for i, data in enumerate([ac_data, coma_data, iac_data, vdn_data, radar_data]):
#     x_list = list(range(len(data)))
#     y = []
#     y.append(np_move_avg(data, 50, mode="same")[0:len(data) - int(len(data)/8)])
#     y.append(np_move_avg(data, 20, mode="same")[0:len(data) - int(len(data)/8)])
#     time = x_list[0:len(data) - int(len(data)/8)]
#     sns.tsplot(time=time, data=y, color=col[i], linestyle='-')
#
# lines = []
# for i, label in enumerate(["ac-qmix", "coma", "iac", "vdn", "radar"]):
#     lines.append(mlines.Line2D([], [], color=col[i], label=label, linestyle='-'))
#
# plt.legend(handles=lines)
# plt.show()


#================= fig1 ========================
plt.figure()
fig, ax = plt.subplots(1, 1)
plt.ticklabel_format(style='sci', scilimits=(0, 0))
plt.xlabel("Train Episode")
plt.ylabel("Average Reward")


for i, data in enumerate([[ddpg_rel_data_step, ddpg_rel_data_score], [ddpg_data_step, ddpg_data_score], [sac_rel_data_step, sac_rel_data_score], [sac_data_step, sac_data_score], [ppo_data_step, ppo_data_score]]):
    x = data[0]
    y = data[1]
    ax.plot(x, y, color=col[i], linestyle='-')
lines = []
for i, label in enumerate(["DDPG-Rel", "DDPG", "SAC-Rel", "SAC", "PPO"]):
    lines.append(mlines.Line2D([], [], color=col[i], label=label, linestyle='-'))
ax.legend(handles=lines)


axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.3, 0.5, 1, 1),
                       bbox_transform=ax.transAxes)
for i, data in enumerate([[ddpg_rel_data_step, ddpg_rel_data_score], [ddpg_data_step, ddpg_data_score], [sac_rel_data_step, sac_rel_data_score], [sac_data_step, sac_data_score], [ppo_data_step, ppo_data_score]]):
    x = data[0]
    y = data[1]
    axins.plot(x, y, col[i], linestyle='-')

axins.set_xlim(2400, 2500)
axins.set_ylim(-150, -125)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)


plt.savefig("fig1.png")
plt.show()

#================= fig 2 ===========================
# label = ["DDPG", "FedDDPG"]
# plt.figure(figsize=(20, 5), dpi=100)
# sns.set(style="whitegrid", font_scale=1)
# for index, datas in enumerate(datas_total):
#     plt.ticklabel_format(style='sci', scilimits=(4, 5), axis='y')
#     plt.subplot(1, 5, index+1)
#     plt.subplots_adjust(hspace=1)
#     legend = []
#     for i, data in enumerate(datas):
#         x = data[0]
#         if True:
#             x = list(range(len(x)))
#         y = data[1]
#         legend.append(label[i])
#         sns.tsplot(time=x, data=y, color=col[i+1], linestyle='-')
#     plt.plot()
#     plt.xlabel("Test Episode", fontsize=10)
#     plt.title(title[index])
#     # if index == 2:
#     if index == 0:
#         plt.ylabel("Test Reward", fontsize=15)
#     line0 = mlines.Line2D([], [], color=col[1], label=label[0], linestyle='-')
#     line1 = mlines.Line2D([], [], color=col[2], label=label[1], linestyle='-')
#     # plt.legend(handles=[line0, line1], ncol=3, loc=9, bbox_to_anchor=(0, 0))
#     plt.legend(handles=[line0, line1])
# plt.savefig("fig2.png")
# plt.show()

# for key in keys:
#     if "train" in key:
#         plt.figure(num=key)
#         data = list(load_dict[key])
#         x_list = list(range(len(data)))
#         y = []
#         y.append(np_move_avg(data, 200, mode="same")[0:len(data) - int(len(data)/8)])
#         y.append(np_move_avg(data, 300, mode="same")[0:len(data) - int(len(data)/8)])
#         time = x_list[0:len(data) - int(len(data)/8)]
#         sns.tsplot(time=time, data=y, color='r', linestyle='-')
#         plt.show()
#     else:
#         plt.figure(num=key)
#         data = list(load_dict[key])
#         x_list = list(range(len(data)))
#         y = []
#         y.append(np_move_avg(data, 10, mode="same")[0:len(data) - int(len(data) / 8)])
#         y.append(np_move_avg(data, 5, mode="same")[0:len(data) - int(len(data) / 8)])
#         time = x_list[0:len(data) - int(len(data) / 8)]
#         sns.tsplot(time=time, data=y, color='r', linestyle='-')
#         plt.show()
