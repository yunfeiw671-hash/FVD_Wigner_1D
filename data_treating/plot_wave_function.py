import os
import numpy as np
import matplotlib.pyplot as plt
#用来画波函数分布图随时间的变化情况

# 基础路径配置
base_path = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\dV=0.06"
dt = 0.01

# 参数配置 ---------------------------------------------------------------
# 修改为一次处理一个beta
current_beta = '2.600'  # 当前要绘制的beta值

# 自定义参数
beta_axis_limits = {
    '1.800': (-0.1, 4),
    '1.400': (-0.1, 2.3),
    '2.600': (-0.1, 5.2)
}

amplify_config = {
    'target_beta': current_beta,  # 确保放大当前beta
    'phi_threshold': 0.41,
    'amplify_factor': 3
}

# 时间步配置
selected_steps = [n for n in range(1500, 4000) if n % 100 == 0]

# 创建输出目录
output_dir = os.path.join(base_path, "combined_plots")
os.makedirs(output_dir, exist_ok=True)

def load_prob_data(beta, step):
    """加载数据并确保场值排序"""
    file_pattern = f"beta={beta}/wave_function_distribution_image/prob_dis_{step}.txt"
    file_path = os.path.join(base_path, file_pattern)
    
    try:
        data = np.loadtxt(file_path, skiprows=1)
        # 按场值排序数据
        sort_idx = np.argsort(data[:, 0])
        phi = data[sort_idx, 0]
        psi2 = data[sort_idx, 1]
        V = data[sort_idx, 2]
        return phi, psi2, V
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None, None

def split_data(phi, psi2, threshold):
    """智能分割数据，避免阈值处连线"""
    split_points = []
    for i in range(len(phi)-1):
        if (phi[i] <= threshold) and (phi[i+1] > threshold):
            split_points.append(i+1)
    
    if not split_points:
        return [(phi, psi2)]
    
    segments = []
    prev = 0
    for sp in split_points:
        segments.append((phi[prev:sp], psi2[prev:sp]))
        prev = sp
    segments.append((phi[prev:], psi2[prev:]))
    return segments

# 主绘图函数（一个beta值）
def plot_single_beta(beta):
    """为单个beta值绘制所有时间步的图像"""
    num_steps = len(selected_steps)
    
    # 根据时间步数量动态计算子图布局
    cols = min(3, num_steps)  # 每行最多4个子图
    rows = (num_steps + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*5), squeeze=False)
    
    # 展平轴数组以便遍历
    axes = axs.flatten()
    
    # 预定义图例句柄
    legend_lines = [
        plt.Line2D([0], [0], color='b', linestyle='-', label=r'$|\psi|^2$'),
        plt.Line2D([0], [0], color='r', linestyle='-', label=r'$V/\Lambda$'),
    ]
    
    if beta == amplify_config['target_beta']:
        legend_lines.append(
            plt.Line2D([0], [0], color='darkblue', linestyle=':', 
                      label=f'Amplified ×{amplify_config["amplify_factor"]}')
        )
    
    for idx, step in enumerate(selected_steps):
        ax = axes[idx]
        phi, psi2, V = load_prob_data(beta, step)
        
        if phi is None:
            continue
        
        # ========== 核心绘图部分 ==========
        # 应用放大操作
        if beta == amplify_config['target_beta']:
            # 分割数据段
            segments = split_data(phi, psi2, amplify_config['phi_threshold'])
            
            # 分别绘制每个数据段
            for seg_phi, seg_psi2 in segments:
                mask = seg_phi > amplify_config['phi_threshold']
                if np.any(mask):
                    # 放大区域使用虚线
                    ax.plot(
                        seg_phi[mask], 
                        seg_psi2[mask] * amplify_config['amplify_factor'],  # 使用seg_psi2
                        color='darkblue', 
                        linestyle=':',
                        linewidth=1.5,
                        alpha=0.9
                    )
                if np.any(~mask):
                    # 正常区域保持实线
                    ax.plot(
                        seg_phi[~mask], 
                        seg_psi2[~mask],  # 使用seg_psi2而不是psi2
                        color='b',
                        linestyle='-'
                    )
        else:
            # 其他beta正常绘制
            ax.plot(phi, psi2, 'b-')
        
        # 绘制势能曲线（使用次坐标轴）
        ax2 = ax.twinx()
        ax2.plot(phi, V, 'r-', alpha=0.7)
        ax2.set_ylim(-0.1, 1)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 设置主坐标轴范围
        ax.set_xlim(-0.65, 1.65)
        ax.set_ylim(beta_axis_limits.get(beta, (-0.1, 4)))
        ax.set_title(f"$t \omega^{{\\star}}$ = {step*dt:.0f}", fontsize=15)
        ax.set_xlabel(r'$\phi/f^{\star}$', fontsize=15)
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(alpha=0.2, linestyle=':')
    
    # 添加全局图例
    fig.legend(handles=legend_lines, loc='lower center', 
              ncol=len(legend_lines), frameon=False, fontsize=15)
    
    # 添加全局标题
    #fig.suptitle(rf"$\beta \omega^{{\star}}$ = {float(beta):.1f}", fontsize=14)
    
    # 隐藏多余的子图
    for j in range(len(selected_steps), len(axes)):
        axes[j].axis('off')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为图例和标题留空间
    
    # 保存并显示
    save_path = os.path.join(output_dir, f'wave_beta_{beta.replace(".", "_")}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# 主执行流程
if __name__ == "__main__":
    # 可以依次处理多个beta，但每次只画一个
    betas_to_plot = ['2.600']  # 只需要当前beta
    # betas_to_plot = ['1.000', '1.800', '2.600']  # 可以添加多个
    
    for beta in betas_to_plot:
        plot_single_beta(beta)