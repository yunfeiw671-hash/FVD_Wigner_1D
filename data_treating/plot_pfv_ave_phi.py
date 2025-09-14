import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy import interpolate
#用来画假真空概率随时间的变化情况和平均场值随时间的变化情况

def interpolate_data(x, y, step=1):  #画avephi-t的时候调用
    """在数据点之间进行线性插值"""
    f = interpolate.interp1d(x, y, kind='linear')
    x_new = np.arange(x[0], x[-1], step)
    y_new = f(x_new)
    return x_new, y_new

# 修正数据加载部分
def load_data(beta):
    gamma_path = os.path.join(file_path, f'beta={beta:.3f}/gamma_values.txt')
    data = np.loadtxt(gamma_path, skiprows=1, usecols=(0, 3, 4, 5, 6))
    time = dt * data[0:, 0]   #slosh time : 0.79
    pfv = data[0:, 1]
    chifv = data[0:, 3]
    return time, pfv, chifv

def plot_pfv_chifv():
    # 动态计算子图行列数 (例如13个图需要4x4布局)
    n_plots = len(betas)
    n_cols = 2 # 每行最多4个子图
    n_rows = int(np.ceil(n_plots / n_cols))  # 计算需要的行数
    
    # 创建自适应子图网格
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axs = axs.flatten()  # 展平为1D数组
    
    # 遍历所有beta值绘制子图
    for i, (beta, data, decay_range) in enumerate(zip(betas, datasets, time_range)):
        ax = axs[i]
        time, pfv, chifv = data
        start_time, end_time = decay_range
        # 绘制曲线
        ax.plot(time, pfv, 'b-', label=r'$P_{FV}$', linewidth=2)
        #ax.plot(time, chifv, 'r--', label=r'$P_{FV}^{\chi}$')
        ax.axvspan(start_time, end_time, color='salmon', alpha=0.35, label='Effective Region')
        # 增大坐标轴刻度字体
        ax.tick_params(axis='both', which='major', labelsize=15)  # 使用子图的tick_params方法
        ax.tick_params(axis='both', which='minor', labelsize=15)  # 次刻度字体稍小

        # 设置坐标轴和标签
        ax.set_xlabel(r'$t\omega^{\star}$', fontsize=15)
        ax.set_title(r'$\beta \omega^{\star}$' f'= {beta:.1f}', y=1.0, fontsize=15)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=15)
        #ax.set_yscale('log')
    
    # 隐藏多余的空子图
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()

def load_avephi(beta):
    gamma_path = os.path.join(file_path, f'beta={beta:.3f}/gamma_values.txt')
    data = np.loadtxt(gamma_path, skiprows=1, usecols=(0, 1))
    
    steps = data[:, 0]
    time = dt * steps
    avephi = data[:, 1]
    
    # 检测实际更新点（值变化的点）
    update_indices = np.where(np.diff(avephi) != 0)[0] + 1
    update_indices = np.insert(update_indices, 0, 0)
    
    # 仅保留实际更新点
    update_time = time[update_indices]
    update_avephi = avephi[update_indices]
    
    # 在更新点之间插值
    time_smooth, avephi_smooth = interpolate_data(update_time, update_avephi, step=0.1*dt)
    
    return time_smooth, avephi_smooth, update_time, update_avephi

def plot_avephi():
    n_plots = len(betas)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axs = axs.flatten()
    
    for i, (beta, data, decay_range) in enumerate(zip(betas, datasets, time_range)):
        ax = axs[i]
        time_smooth, pfv_smooth, update_time, update_pfv = data
        start_time, end_time = decay_range
        
        # 绘制平滑曲线
        ax.plot(time_smooth, pfv_smooth, 'b-', label=r'$<\phi>$', linewidth=2)
        
        # 标出实际数据点
        #ax.scatter(update_time, update_pfv, color='blue', s=20, alpha=0.7, zorder=5)
        
        # 标记有效区域
        #ax.axvspan(start_time, end_time, color='salmon', alpha=0.35, label='Effective Region')
        
        # 设置图表属性
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel(r'$t\omega^{\star}$', fontsize=15)
        ax.set_ylabel(r'$\left <\phi/f^{\star} \right >$', fontsize=15)
        ax.set_title(r'$\beta \omega^{\star}$' f'= {beta:.1f}', fontsize=15)
        ax.grid(True, alpha=0.3)
        #ax.legend(loc='best', fontsize=15)
        
        # 设置合适的Y轴范围
        y_min = min(pfv_smooth) * 0.95 if min(pfv_smooth) > 0 else min(pfv_smooth) * 1.05
        y_max = max(pfv_smooth) * 1.05
        ax.set_ylim(y_min, y_max)
    
    # 隐藏多余的空子图
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    #plt.savefig('smoothed_pfv_plot.png', dpi=300)
    plt.show()


dt = 0.01
file_path = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\dV=0.06"

# 加载四个beta的数据集
#betas = [2.000, 2.200, 2.400, 2.600, 2.800, 3.000]
#step_range = [(332, 431), (343, 442), (354, 453), (382, 481), (374, 473), (386, 485)]   #frac = 1.0
#step_range = [(311, 460), (320, 469), (331, 480), (356, 505), (352, 501), (353, 501)]   #frac = 1.5

#betas = [1.400, 1.800, 2.600, 3.000]
#step_range = [(150, 500), (150, 1000), (1499, 3999), (1499, 3999)] 

betas = [1.400, 2.600]
step_range = [(150, 500), (1499, 3999)] 
time_range = [((start*dt), (end*dt)) for start, end in step_range]


#datasets = [load_avephi(beta) for beta in betas]   #画平均场值变化
#plot_avephi()

datasets = [load_data(beta) for beta in betas]    #画概率变化
plot_pfv_chifv()



