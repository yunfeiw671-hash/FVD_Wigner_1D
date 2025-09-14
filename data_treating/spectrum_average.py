import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
#用来画有效区间内系综平均（样本平均+时间平均）后的功率谱，有效区间通过extract_gamma.py运行后得到

# 参数定义
a = 0.8 
b = -1.8400290793253238 
c = 0.9800221080139118
N_x = 1024
dx = 0.25
Half_Nx = int(N_x/2 + 1)

def calculate_beta(power_spectrum):
    """计算场涨落的方差β = ⟨δφ²⟩（保持原样）"""
    if N_x % 2 == 0:
        sum_terms = power_spectrum[0] + 2 * np.sum(power_spectrum[1:-1]) + power_spectrum[-1]
    else:
        sum_terms = power_spectrum[0] + 2 * np.sum(power_spectrum[1:])
    return sum_terms

# 定义瑞利-金斯分布函数
def rayleigh_jeans_phi(k, T_eff, m_eff):
    """瑞利-金斯分布公式 for φ field"""
    return T_eff / (k**2 + m_eff**2)

def rayleigh_jeans_pi(k, T_eff):
    """瑞利-金斯分布公式 for Π field (constant)"""
    return np.ones_like(k) * T_eff

def fit_pi_spectrum(avg_spectrum, k_values, num_fit_points=10):
    """拟合Π场功率谱，使用前10个低k值"""
    # 选择低k值点进行拟合（排除k=0点，从第1个到第10个点）
    low_k_indices = np.argsort(k_values)[1:1+num_fit_points]  # 取前10个最小非零k值
    low_k = k_values[low_k_indices]
    low_spectrum = avg_spectrum[low_k_indices]
    
    # 计算平均作为T_eff的估计
    T_eff = np.mean(low_spectrum)
    return T_eff

def fit_phi_spectrum(avg_spectrum, k_values, T_eff, num_fit_points=20):
    """拟合φ场功率谱，使用Teff和低k值点拟合m_eff"""
    # 选择低k值点进行拟合（排除k=0点）
    low_k_indices = np.argsort(k_values)[1:1+num_fit_points]  # 取前20个最小非零k值
    low_k = k_values[low_k_indices]
    low_spectrum = avg_spectrum[low_k_indices]
    
    # 使用曲线拟合得到m_eff
    popt, _ = curve_fit(lambda k, m_eff: rayleigh_jeans_phi(k, T_eff, m_eff), 
                       low_k, low_spectrum, p0=[1.0])
    m_eff = popt[0]
    return m_eff

def plot_spectrum_with_fit(spectrum_dir, prefix, time_range, color, marker,
                          uv_cutoff=None, field_label=None, field_type='phi',
                          T_eff=None, num_fit_points=10):
    """
    绘制单个场的功率谱及其瑞利-金斯拟合
    参数：
        prefix: 文件前缀 ('phi_spectrum_step' 或 'pi_spectrum_step')
        field_label: 场类型标签 ('\delta\phi' or '\delta\Pi')
        field_type: 'phi' 或 'pi'
        T_eff: 有效温度（仅phi场需要）
    """
    # 获取文件列表
    all_files = sorted([f for f in os.listdir(spectrum_dir) 
                      if f.startswith(prefix)])
    
    # 筛选时间范围内的文件
    valid_files = []
    steps = []
    for f in all_files:
        step = int(f.split('_')[-1].split('.')[0])
        if time_range[0] <= step <= time_range[1]:
            valid_files.append(f)
            steps.append(step)
    
    if not valid_files:
        print(f"Warning: No data for prefix {prefix} in time range {time_range}")
        return None
    
    # 读取并平均数据
    spectra = []
    k_values = None
    for f in valid_files:
        data = np.loadtxt(os.path.join(spectrum_dir, f))
        k = data[:, 0]
        spectrum = data[:, 1] / (N_x**2)  # 归一化
        
        if k_values is None:
            k_values = k
        spectra.append(spectrum)
    
    # 计算时间平均
    avg_spectrum = np.mean(spectra, axis=0)
    
    # 计算β值（可选）
    beta = calculate_beta(avg_spectrum)
    print(f"[{prefix}] β = {beta:.6f}")
    
    # 创建新图形
    plt.figure(figsize=(15, 9))
    ax = plt.gca()
    
    # 绘制模拟结果（散点图）
    ax.scatter(k_values, avg_spectrum, 
              c=color,
              marker=marker,
              s=80,           # 点大小
              edgecolors='k',  # 点边缘颜色
              alpha=0.7,
              linewidths=0.8,
              label='Simulation') 
    
    # 添加拟合曲线
    if field_type == 'pi':
        # 拟合Π场
        T_eff = fit_pi_spectrum(avg_spectrum, k_values, num_fit_points)
        fit_curve = rayleigh_jeans_pi(k_values, T_eff)
        ax.plot(k_values, fit_curve, 'r-', linewidth=2.5, label='Fit')
        # 添加低k区域拟合点标记（仅用于可视化）
        low_k_indices = np.argsort(k_values)[1:1+num_fit_points]
        ax.scatter(k_values[low_k_indices], avg_spectrum[low_k_indices], 
                  facecolors='none', edgecolors='g', s=120, linewidths=1.5,
                  zorder=10)
    else:  # phi场
        if T_eff is None:
            raise ValueError("T_eff must be provided for phi field fitting")
        # 拟合φ场
        m_eff = fit_phi_spectrum(avg_spectrum, k_values, T_eff)
        fit_curve = rayleigh_jeans_phi(k_values, T_eff, m_eff)
        ax.plot(k_values, fit_curve, 'r-', linewidth=2.5, label='Fit')
    
    # 添加紫外截断线
    if uv_cutoff is None:
        uv_cutoff = np.pi / dx
    ax.axvline(uv_cutoff, color='black', linestyle='--', alpha=0.6)
    ax.text(uv_cutoff * 1.1, ax.get_ylim()[1] * 0.1, 
           r'$\pi/\Delta x$', 
           rotation=90, ha='center', fontsize=20)
    
    # 图表装饰
    ax.set_xlabel('Wave Number k', fontsize=25)
    ax.set_ylabel(rf'Power Spectrum ${field_label}$', fontsize=25)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.gca().yaxis.get_offset_text().set_fontsize(25)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    #plt.ylim(0,0.025)
    # 添加图例
    ax.legend(loc='lower left', fontsize=16, frameon=True, shadow=True, fancybox=True)
    
    #title = rf'Time-Averaged Power Spectrum: ${field_label}$'
    #plt.title(title, fontsize=20, pad=15)
    
    # 设置对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 保存结果
    output_name = f"average_{prefix}_fit_{time_range[0]}-{time_range[1]}.pdf"
    #output_path = os.path.join(spectrum_dir, output_name)
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return T_eff if field_type == 'pi' else None

# 使用示例
if __name__ == "__main__":
    spectrum_path = r'E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\dV=0.06\beta=2.600\spectrum'
    #time_range = (213, 463)   #dV=0.06 beta=1.4
    time_range = (2499, 3999)   #dV=0.06 beta=2.6
    uv_cutoff = np.pi / dx  # 计算紫外截断

    # 先处理Π场获得T_eff
    T_eff = plot_spectrum_with_fit(
        spectrum_path, 
        'pi_spectrum_step',
        time_range,
        color='red',
        marker='s',
        uv_cutoff=uv_cutoff,
        field_label=r'<|\delta \Pi_k|^2>',
        field_type='pi'
    )

    #使用从Π场获得的T_eff处理φ场
    plot_spectrum_with_fit(
        spectrum_path, 
        'phi_spectrum_step',
        time_range,
        color='blue',
        marker='o',
        uv_cutoff=uv_cutoff,
        field_label=r'<|\delta \phi_k|^2>',
        field_type='phi',
        T_eff=T_eff
    )