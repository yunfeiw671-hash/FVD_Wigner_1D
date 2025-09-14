import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
#样本平均后功率谱的随时间变化情况

def plot_spectrum_evolution(spectrum_dir, prefix="phi_spectrum_step", cmap='viridis'):
    # 获取所有频谱文件
    files = sorted([f for f in os.listdir(spectrum_dir) if f.startswith(prefix)])
    
    # 提取时间步信息并排序
    steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
    min_step, max_step = min(steps), max(steps)
    step_interval=100
    
    # 筛选每隔step_interval步的文件
    selected_indices = []
    for i, step in enumerate(steps):
        if step % step_interval == 0:  # 每隔step_interval步选择一次
            selected_indices.append(i)
    
    selected_files = [files[i] for i in selected_indices]
    selected_steps = [steps[i] for i in selected_indices]
    
    # 计算对应的时间（秒）
    times = [step * dt for step in selected_steps]
    min_time, max_time = min(times), max(times)
    
    output_path = os.path.join(spectrum_dir, 'phi_spectrum_evolution.png')
   
    # 准备绘图
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 获取颜色映射
    colormap = plt.get_cmap(cmap)
    
    # 标准化时间步到[0,1]区间
    time_norm = Normalize(vmin=min_time, vmax=max_time)
    
    # 遍历选中的文件绘制频谱
    for i, (f, step) in enumerate(zip(selected_files, selected_steps)):
        data = np.loadtxt(os.path.join(spectrum_dir, f))
        k = data[:, 0]
        spectrum = data[:, 1]
        
        spectrum = spectrum / (N_x**2)  # 归一化
        
        # 计算颜色（基于时间）
        color = colormap(time_norm(step * dt))
        
        plt.semilogx(k, spectrum, color=color, lw=1.2, alpha=0.85)

    # 添加紫外截断标记
    plt.axvline(uv_cutoff, color='r', linestyle='--', linewidth=1, alpha=0.7)
    #plt.text(uv_cutoff*1.1, 1e-4, r'$\pi/\Delta x$', color='red', rotation=90)

    # 添加颜色条（基于时间）
    sm = ScalarMappable(cmap=colormap, norm=time_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time', rotation=270, labelpad=15, fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    # 设置刻度间隔（基于时间）
    time_ticks = np.linspace(min_time, max_time, 5)  # 5个等间隔时间点
    cbar.set_ticks(time_ticks)
    cbar.set_ticklabels([f"{t:.0f}" for t in time_ticks])  # 格式化为两位小数
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    # 图表装饰
    plt.xlabel('Wave Number k', fontsize=17)
    plt.ylabel(r'Power Spectrum $|\delta \phi_k|^2$', fontsize=17)
    plt.yscale('log')
    plt.xticks(fontsize=19)  
    plt.yticks(fontsize=19)
    #plt.title('Spectral Evolution with Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(k.min(), k.max())
    #plt.ylim(0,0.034)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # 保存图像
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# 使用示例
# 紫外截断参数
dx = 0.25
dt=0.01
uv_cutoff = np.pi / dx
N_x =1024
#spectrum_file = r'E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\spectrum\Nx=1024_dx=0.25\dV=0.060\beta=1.400\spectrum'
spectrum_file = r'E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\dV=0.06\beta=1.400\spectrum'
plot_spectrum_evolution(spectrum_file, prefix='phi_spectrum_step', cmap='plasma')



