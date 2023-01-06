# Autonomous-Parking-Narrow-Space

    Run Code:
    python main.py --path_num 7 --exp_name test 

目前适用于障碍物为四边形的情况，因为在计算穿透面积的时候需要先预定义好矩阵的大小，因此当障碍物的
形状发生改变时，无法使用原有的矩阵求解。