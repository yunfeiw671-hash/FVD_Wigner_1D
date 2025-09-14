import numpy as np  #新的算法，在每个样本上统计满足条件的空间点（以前是在每个空间点处统计满足条件的样本）
import matplotlib.pyplot as plt 
import matplotlib 
import numba
import time
import os
import argparse
import cmath
import plotly.graph_objects as go
from mpmath import mp, exp, log
from scipy.optimize import fsolve,minimize
from scipy.sparse import diags
matplotlib.use('Agg')
mp.dps = 10
#numba.set_num_threads(64)
#print("Numba maximum threads:", numba.get_num_threads()) 
#os.environ['OPENBLAS_NUM_THREADS']='64'
#print("OPENBLAS_NUM_THREADS:", os.environ.get('OPENBLAS_NUM_THREADS'))

def write_param():
    with open(parameter_file, "w") as file:
        file.write("# Program parameters\n")
        file.write(f"N_samples = {N_samples}  # Number of phi and pi field samples\n")
        file.write(f"dx = {dx}  # Spatial step size\n")
        file.write(f"dt = {dt}  # Time step size\n")
        file.write(f"N_steps = {N_steps}  # Number of time steps\n")
        file.write(f"N_x = {N_x}  # Spatial grid size\n")
        file.write(f"length_x = {length_x} # Dimensionless spatial length\n\n")

        file.write("# Potential parameters V = a*phi**2 + b*phi**3 + c*phi**4\n")
        file.write(f"a = {a} \n")
        file.write(f"b = {b} \n")
        file.write(f"c = {c} \n")
        file.write(f"phi_l = {phi_l}  # False vacuum level point\n")
        file.write(f"(phi_m,V_max) = ({phi_m},{V_max})  # Potential maximum point\n")
        file.write(f"phi_n = {phi_n}  # True vacuum level point\n")
        file.write(f"phi_m1 = {phi_m1}  # False vacuum point\n")
        file.write(f"(phi_m2,V_min) = ({phi_m2},{V_min})  # True vacuum point\n\n")
        

        file.write("# Physical parameters\n")
        file.write(f"beta = {beta}  # Dimensionless Inverse temperature (1/T)\n")
        file.write(f"m_eff = {m_eff}  # Dimensionless Effective mass\n")
        file.write(f"a_f = {a_f}  # Scale factor of comoving coordinates\n")
        file.write(f"erfa = {erfa}  \n")
        file.write(f"wstar = {wstar}  # Energy scale factor for frequency\n")
        file.write(f"fstar = {fstar}  # Energy scale factor for field\n")
        file.write(f"R = {R}  # Dimensionless radius factor\n")
        file.write(f"lamda = {lamda}  \n")

#波函数分布图
def save_Psi_squer(phi_pairs, pi_pairs, W, step):
    prob_values = []
    dphi = phi_intervals[1]-phi_intervals[0]
    
    # 计算概率分布
    for i in range(N_intervals):
        phi_1 = phi_intervals[i]
        phi_2 = phi_intervals[i + 1]

        Kai = compute_Kai(phi_pairs, pi_pairs, phi_1, phi_2)
        prob = compute_prob(W, Kai)/dphi  
        prob_values.append(prob)

    # 准备保存数据
    interval_centers = (phi_intervals[:-1] + phi_intervals[1:]) / 2
    V_values = [V(phi) for phi in interval_centers]  # 计算每个区间中心对应的势能
    
    # 将数据保存为txt文件（三列：区间中心值，概率值，势能值）
    data_to_save = np.column_stack((interval_centers, prob_values, V_values))
    np.savetxt(f"./wave_function_distribution_image/prob_dis_{step}.txt", 
              data_to_save,
              header="phi_center |Psi|^2 V(phi)",
              comments='',
              fmt='%.16f')

# W分布图
def save_W_distribution_plots(W_values, step):  
    #Wmin = 3
    #Wmax = 13
    plt.figure(figsize=(12, 10))
    #plt.imshow(W_values, origin='lower', cmap='viridis', vmin=Wmin, vmax=Wmax)
    plt.imshow(W_values, origin='lower', cmap='viridis')
    plt.colorbar(label="W Values")
    plt.title(f"W Distribution at Time={step}")
    plt.savefig(f'./W_image/W_distribution_{step}.png')
    plt.close()

#W值分布的直方图      
def save_W_histogram(W_values, step):
    
    plt.figure(figsize=(10, 6))
    # 创建直方图
    #n, bins = np.histogram(W_values.flatten(), bins=100, density=True)  # 归一化为概率密度
    
    # 添加统计信息
    plt.text(0.95, 0.95, 
            f'Mean: {np.mean(W_values):.2e}\nStd: {np.std(W_values):.2e}',
            transform=plt.gca().transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8))
            
    W_flat = W_values.flatten()
    plt.hist(W_flat, bins=100, density=True, alpha=0.7, color='b')
    plt.xlim([np.min(W_flat)-0.000001, np.max(W_flat)+0.000001])  # 设置 x 轴的范围
    plt.yscale('log')
    plt.xlabel('W Value')
    #plt.ylabel('Probability Density')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'./W_image/W_histogram_{step}.pdf', bbox_inches='tight')
    plt.close()

#样本平均场的分布图
def plot_sample_average_field(phi_pairs, pi_pairs, step):
    phi_ylim=(-0.5, 1.5)
    pi_ylim=(-2, 2)
    fig, axs = plt.subplots(2,1 , figsize=(12, 12))
    #对所有样本取平均
    phi_avg=np.mean(phi_pairs,axis=(0, 1))
    pi_avg=np.mean(pi_pairs,axis=(0, 1))

    axs[0].plot(phi_avg, label=f"Average Phi, Time={step}", color='blue')
    axs[0].set_title(f"1D Phi Distribution, Time={step}")
    axs[0].set_xlabel("Position Index")
    axs[0].set_ylabel("Phi Value")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylim(phi_ylim)
 
    axs[1].plot(pi_avg, label=f"Average Pi, Time={step}", color='green')
    axs[1].set_title(f"1D Pi Distribution, Time={step}")
    axs[1].set_xlabel("Position Index")
    axs[1].set_ylabel("Pi Value")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim(pi_ylim)
    
    plt.tight_layout()
    plt.savefig(f'./field_image/Field_sample_average_{step}.png')
    plt.close()

def save_average_field(phi_pairs, pi_pairs, phi_filename="avephidis.txt", pi_filename="avepidis.txt"):
    phi_avg=np.mean(phi_pairs,axis=(0, 1))
    pi_avg=np.mean(pi_pairs,axis=(0, 1))
    with open(phi_filename, "a") as f_phi, open(pi_filename, "a") as f_pi:
        np.savetxt(f_phi, phi_avg[None], fmt="%.6f", delimiter=" ")
        np.savetxt(f_pi, pi_avg[None], fmt="%.6f", delimiter=" ")

def save_pair_field(phi_array, pi_array, pair, phi_filename=None, pi_filename=None):
    if phi_filename is None or pi_filename is None:
        phi_filename = f"./field_pairdis/phidis({pair[0]},{pair[1]}).txt"
        pi_filename  = f"./field_pairdis/pidis({pair[0]},{pair[1]}).txt"
    with open(phi_filename, "a") as f_phi, open(pi_filename, "a") as f_pi:
        np.savetxt(f_phi, phi_array[None], fmt="%.6f", delimiter=" ")
        np.savetxt(f_pi, pi_array[None], fmt="%.6f", delimiter=" ")

#某个样本对的场分布
def plot_sample_pair_field(phi_pairs, pi_pairs, step):
    fig, axs = plt.subplots(2,2 , figsize=(20, 16))
    i=6    #样本对标记
    j=7
    phi_1 = phi_pairs[i,i]  #对角线的
    pi_1 = pi_pairs[i,i]
    phi_2 = phi_pairs[i,j]  #非对角线的
    pi_2 = pi_pairs[j,i]
    phi_ylim=(-0.5, 1.5)
    pi_ylim=(-2, 2)

    axs[0, 0].plot(phi_1, label=f"phi_{i}_{i}, Time={step}", color='blue')
    axs[0, 0].set_title(f"phi_{i}_{i} Distribution, Time={step}")
    axs[0, 0].set_xlabel("Position Index")
    axs[0, 0].set_ylabel("Phi Value")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_ylim(phi_ylim)

    axs[0, 1].plot(pi_1, label=f"pi_{i}_{i}, Time={step}", color='green')
    axs[0, 1].set_title(f"pi_{i}_{i} Distribution, Time={step}")
    axs[0, 1].set_xlabel("Position Index")
    axs[0, 1].set_ylabel("Pi Value")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim(pi_ylim)

    axs[1, 0].plot(phi_2, label=f"phi_{i}_{j}, Time={step}", color='purple')
    axs[1, 0].set_title(f"phi_{i}_{j} Distribution, Time={step}")
    axs[1, 0].set_xlabel("Position Index")
    axs[1, 0].set_ylabel("Phi Value")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim(phi_ylim)

    axs[1, 1].plot(pi_2, label=f"pi_{j}_{i}, Time={step}", color='red')
    axs[1, 1].set_title(f"pi_{j}_{i} Distribution, Time={step}")
    axs[1, 1].set_xlabel("Position Index")
    axs[1, 1].set_ylabel("Pi Value")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(pi_ylim)

    plt.tight_layout()
    plt.savefig(f'./field_image/field_sample_pairs_{step}.png')
    plt.close()

#某个点处的不同样本值分布
def save_field_point_distributions(phi_pairs, pi_pairs, points, output_dir='./field_image', bins=50):  
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    for idx, point in enumerate(points):
        x = point

        # 提取固定空间点处的样本值
        phi_values = phi_pairs[:, :, x].flatten()
        pi_values = pi_pairs[:, :, x].flatten()

        # 创建图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # 绘制 phi 场分布
        axes[0].hist(phi_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title(f"$\\phi$ Distribution at Point ({x})")
        axes[0].set_xlabel("$\\phi$")
        axes[0].set_ylabel("Frequency")

        # 绘制 pi 场分布
        axes[1].hist(pi_values, bins=bins, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title(f"$\\pi$ Distribution at Point ({x})")
        axes[1].set_xlabel("$\\pi$")

        # 保存图像
        file_path = os.path.join(output_dir, f'Field_distribution_point_{idx}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        print(f"Saved plot for point {point} to {file_path}")

#|phi_k|^2 的频谱分布
def save_field_in_kspace_spectrum(phi_samples, phi_trda2_k, k_values, dx, key):
    """
    绘制理论功率谱和模拟生成的初始功率谱对比图  
    参数:
    phi_samples -- 模拟生成的phi场样本，形状为 (N_samples, N_x)
    phi_trda2_k -- 理论功率谱，形状为 (N_x//2+1) 对应 rfft 频率
    k_values    -- 理论频率数组（对应于 phi_trda2_k 的 k 值）
    """
    N_samples, N_x = phi_samples.shape
    
    # 1. 计算模拟样本的平均功率谱
    phi_k_mag_sum = np.zeros(N_x//2+1, dtype=np.float64)
    
    for phi in phi_samples:
        phi_k = np.fft.rfft(phi)
        phi_k_mag_sum += np.abs(phi_k)**2
    
    simulated_spectrum = phi_k_mag_sum / N_samples
    
    # 2. 绘制理论功率谱和模拟功率谱对比图
    plt.figure(figsize=(10, 8))
    
    # 理论功率谱（理论功率谱已包含体积因子）
    plt.plot(k_values, phi_trda2_k, 
             'r-', lw=2, alpha=0.7, label='Theoretical Spectrum')
    
    # 模拟功率谱（点图）
    plt.scatter(k_values, simulated_spectrum, 
                s=15, alpha=0.7, marker='o', edgecolor='b', 

                facecolor='none', label='Simulated Initial Spectrum')
    
    # 添加Nyquist频率线
    plt.axvline(np.pi / dx, color='g', linestyle='--', alpha=0.5, label='Nyquist Frequency')
    
    # 设置坐标轴
    plt.xlabel(r"Wave number $k$", fontsize=12)
    plt.ylabel(key + f'Power Spectrum', fontsize=12)
    #plt.title(key + f'Theoretical vs Simulated Power Spectrum', fontsize=14)
    
    # 双对数坐标
    plt.xscale('log')
    plt.yscale('log')
    
    # 网格和图例
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(key + f'_spectrum_comparison.png', dpi=300)
    plt.close()
    print("Saved spectrum comparison plot and data.")

#计算并保存所有样本对的平均频谱
def save_spectrum(phi_pairs, pi_pairs, step, dx, N_samples):
    N_x = phi_pairs.shape[2]
    # 初始化 rfft 频谱数组
    N_freqs = N_x // 2 + 1  # rfft 的频率点数
    phi_spectrum_sum = np.zeros(N_freqs, dtype=np.float64)
    pi_spectrum_sum = np.zeros(N_freqs, dtype=np.float64)
    # 遍历所有样本对
    for i in range(N_samples):
        for j in range(N_samples):
            phi = phi_pairs[i, j]
            pi = pi_pairs[i, j]
            
            # 计算傅里叶变换的模平方
            phi_spectrum = np.abs(np.fft.rfft(phi))**2
            pi_spectrum = np.abs(np.fft.rfft(pi))**2
            
            # 累加频谱
            phi_spectrum_sum += phi_spectrum
            pi_spectrum_sum += pi_spectrum
    
    # 计算平均频谱
    phi_avg = phi_spectrum_sum / N_samples**2
    pi_avg = pi_spectrum_sum / N_samples**2
    
    # 生成对应的k值（非负频率，不需要 shift）
    k_values = np.fft.rfftfreq(N_x, dx) * 2 * np.pi
    
    os.makedirs('spectrum', exist_ok=True)
    
    # 保存phi频谱
    phi_filename = f"spectrum/phi_spectrum_step_{step:04d}.txt"
    np.savetxt(phi_filename, np.column_stack((k_values, phi_avg)),
               header="k_value |phi_k|^2", fmt="%.6e")
    
    # 保存pi频谱
    pi_filename = f"spectrum/pi_spectrum_step_{step:04d}.txt"
    np.savetxt(pi_filename, np.column_stack((k_values, pi_avg)),
               header="k_value |pi_k|^2", fmt="%.6e")

#绘制场值的直方图，查看样本的场值分布
def plot_fieldvalues(field, key):
    field_flat = field.flatten()
    plt.hist(field_flat, bins=100, alpha=0.7, color='b', label=key+' Distrubion')
    plt.xlim([np.min(field_flat), np.max(field_flat)])  # 设置 x 轴的范围
    plt.legend()
    plt.xlabel(key)
    plt.ylabel("Frequency")    
    plt.savefig(key + f' Value Distrubion')
    plt.close()

def plot_save_fieldvalues(field, key, filename="field_distribution.txt"):
    counts, bins = np.histogram(field.flatten(), bins=100, density=True)

    # 计算每个区间的中点场值
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 保存数据
    np.savetxt(filename, 
              np.column_stack((bin_centers, counts)),
              fmt='%.6f %.9f',
              header="FieldValue Frequency")
              
    field_flat = field.flatten()
    plt.hist(field_flat, bins=100, alpha=0.7, color='b', label=key+' Distrubion')
    plt.xlim([np.min(field_flat), np.max(field_flat)])  # 设置 x 轴的范围
    plt.legend()
    plt.xlabel(key)
    plt.ylabel("Frequency")    
    plt.savefig(key + f' Value Distrubion')
    plt.close()

#势能图
def plot_potential():
    phi_range = np.linspace(-0.5, 1.5, 100)
    V_values = V(phi_range)
    plt.plot(phi_range, V_values, label=fr'$V(\phi) = {a:.3f}\phi^2  {b:.3f}\phi^3 + {c:.3f}\phi^4$')
    plt.scatter(phi_m, V_max, color='red', zorder=5)
    plt.scatter(phi_m2, V_min, color='blue', zorder=5)
    plt.text(phi_m, V_max + 0.02, f'Barrier\n({phi_m:.3f}, {V_max:.3f})', 
             ha='center', fontsize=8, color='red')
    plt.text(phi_m2, V_min - 0.2, f'Vacuum\n({phi_m2:.3f}, {V_min:.3f})', 
             ha='center', fontsize=8, color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # 添加 x 轴
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # 添加 y 轴
    plt.xlabel(r'$\phi$', fontsize=14)
    plt.ylabel(r'$V(\phi)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('potential_function.png')  
    plt.close() 

def E(phi,pi):
    kinetic = 0.5 * np.sum(pi**2)
    grad_x = finite_diff_gradient(phi)
    gradient = 0.5 * np.sum(grad_x**2) 
    potential = np.sum(V(phi))
    T = dx * (kinetic + gradient)
    Hi = dx * np.sum(Vi(phi))
    E = dx * (kinetic + gradient + potential)
    return T, Hi, E

#计算能量，检查时间演化过程中的能量守恒
def average_energy(phi_pairs, pi_pairs):
    total_T = 0
    total_E = 0
    total_Hi = 0
    for i in range(N_samples):
        for j in numba.prange(N_samples):
            T_val, Hi_val, E_val = E(phi_pairs[i,j], pi_pairs[i,j])
            total_T += T_val
            total_Hi += Hi_val
            total_E += E_val     
    average_T = total_T/N_samples**2
    average_Hi = total_Hi/N_samples**2
    average_E = total_E/N_samples**2
    return average_T, average_Hi, average_E

#定义势能
@numba.njit(fastmath=True, nogil = True)
def V(phi):
    return a*phi**2 + b*phi**3 + c*phi**4

@numba.njit(fastmath=True, nogil = True)
def Vi(phi):   #非自由部分势能
    return b*phi**3 + c*phi**4

# 定义势能的导数
@numba.njit(fastmath=True, nogil = True)
def V_prime(phi):
    return 2*a*phi + 3*b*phi**2 + 4*c*phi**3

@numba.njit(fastmath=True, nogil = True)
def V_2prime(phi):
    return 2*a + 6*b*phi + 12*c*phi**2

# 找到极值点，包括极大值点phi_m和极小值点phi_m1,phi_m2
def find_extrema():
    def f_prime(phi):
        return V_prime(phi)

    roots = []
    for x0 in np.linspace(-2, 5, 100): 
        root = fsolve(f_prime, x0, maxfev=100)[0]
        if not any(abs(root - r) < 1e-5 for r in roots):  
            roots.append(root)

    return sorted(roots)

# 找到phi_l,  phi_n（V(phi_l)=V(phi_m)=V(phi_n)）
def find_special_points():
    # 1. 找到所有的极值点（通过V'(phi) = 0）
    extrema = find_extrema()
    print("Found extrema:", extrema)  # 调试输出

    if len(extrema) < 3:
        raise ValueError("未能找到足够的极值点。需要至少三个极值点。")

    phi_m = extrema[1]  
    V_m = V(phi_m)  

    # 3. 筛选出两个极小值点 phi_m1 和 phi_m2
    phi_m1 = max([ext for ext in extrema if ext < phi_m])  # 取 phi_m 左边的极小值点
    phi_m2 = min([ext for ext in extrema if ext > phi_m])  # 取 phi_m 右边的极小值点

    # 4. 找到在 V(phi) = V_m 时的两个解：phi_l 和 phi_n
    def V_eq_target(phi):
        return V(phi) - V_m

    phi_l = fsolve(V_eq_target, phi_m1 - 5, maxfev=1000)[0]  # 在 phi_m1 左侧寻找 phi_l
    phi_n = fsolve(V_eq_target, phi_m2 + 5, maxfev=1000)[0]  # 在 phi_m2 右侧寻找 phi_n

    return phi_l, phi_m1, phi_m, phi_m2, phi_n

# 为了用jit优化， 手动算gradient
@numba.njit(fastmath=True, nogil = True)
def finite_diff_gradient(phi):
    grad_x = np.zeros_like(phi)
    grad_x[1:-1] = (phi[2:] - phi[:-2]) / (2 * dx)  # 中间部分使用中心差分
    grad_x[0] = (phi[1] - phi[-1]) / dx  # 边界使用一阶差分
    grad_x[-1] = (phi[0] - phi[-2]) / dx

    return grad_x

@numba.njit(fastmath=True, nogil = True)
def laplacian_1d(phi):
    # 获取矩阵的形状
    nx = grid_shape
    # 初始化拉普拉斯矩阵
    laplacian_phi = np.zeros_like(phi)
    # 周期性边界条件的拉普拉斯算符
    laplacian_phi[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    
    # 处理边界上的拉普拉斯，使用周期性边界条件
    laplacian_phi[0] = (phi[1] - 2 * phi[0] + phi[-1]) / dx**2
    laplacian_phi[-1] = (phi[0] - 2 * phi[-1] + phi[-2]) / dx**2
    
    return laplacian_phi

# 加个框
@numba.njit()
def print_boxed(text, char='*', width=50):

    border = char * width
    print(border)
    print(f"{char} {text.center(width - 4)} {char}")
    print(border)
############################################################

@numba.njit(fastmath=True, parallel=True, nogil = True)
def substep(phi_pairs, pi_pairs):   #四阶龙格库塔算法

    def dphi_dt(phi, pi):
        return pi   
    def dpi_dt(phi, pi):
        laplacian_phi = laplacian_1d(phi)
        return laplacian_phi - (2*a*phi+ 3*b*phi**2+ 4*c*phi**3)
    print("Sample\t\tTotal Samples\t\tTime Cost")
    for i in range(N_samples):
        for j in numba.prange(N_samples):
            # 更新样本对
            phi_cur = phi_pairs[i, j]
            pi_cur = pi_pairs[i, j]

            k1_phi = dphi_dt(phi_cur, pi_cur) * dt
            k1_pi = dpi_dt(phi_cur, pi_cur) * dt

            k2_phi = dphi_dt(phi_cur + 0.5 * k1_phi, pi_cur + 0.5 * k1_pi) * dt
            k2_pi = dpi_dt(phi_cur + 0.5 * k1_phi, pi_cur + 0.5 * k1_pi) * dt

            k3_phi = dphi_dt(phi_cur + 0.5 * k2_phi, pi_cur + 0.5 * k2_pi) * dt
            k3_pi = dpi_dt(phi_cur + 0.5 * k2_phi, pi_cur + 0.5 * k2_pi) * dt

            k4_phi = dphi_dt(phi_cur + k3_phi, pi_cur + k3_pi) * dt
            k4_pi = dpi_dt(phi_cur + k3_phi, pi_cur + k3_pi) * dt

            phi_pairs[i, j] = phi_cur + (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi) / 6.0
            pi_pairs[i, j] = pi_cur + (k1_pi + 2 * k2_pi + 2 * k3_pi + k4_pi) / 6.0
        if i % 10 == 0:
            print(f"{i}\t\t{N_samples}")
    return phi_pairs, pi_pairs

@numba.njit(fastmath=True, parallel=True, nogil=True)
def substep2(phi_pairs, pi_pairs):   #蛙跳法
    print("Sample\t\tTotal Samples\t\tTime Cost")
    # 中间步骤的更新
    for i in range(N_samples):
        for j in numba.prange(N_samples):
            # 当前的 phi 和 pi
            phi_cur = phi_pairs[i, j]
            pi_cur = - pi_pairs[i, j]  #取负号是为了实现时间反向演化
            # 半步更新 pi
            pi_half = pi_cur + 0.5 * dt * (laplacian_1d(phi_cur) - V_prime(phi_cur))
            # 一步更新 phi
            phi_new = phi_cur + dt * pi_half
            # 半步更新 pi
            pi_new = pi_half + 0.5 * dt * (laplacian_1d(phi_new) - V_prime(phi_new))
            # 更新样本对
            phi_pairs[i, j] = phi_new
            pi_pairs[i, j] = - pi_new   #再次取负号是为了恢复到正时背景

        # 打印进度
        if i % 100 == 0:
            print(f"{i}\t\t{N_samples}")
    
    return phi_pairs, pi_pairs

# 定义函数用于生成随机样本 处理一维频域对称性）phi_trda2_k, pi_trda2_k: 形状为 (N//2+1) 的功率谱数组
def generate_initial_samples_corrected(phi_trda2_k, pi_trda2_k, N):

    # 初始化频域数组（rfft格式，仅非负频率）
    phi_k = np.zeros(N//2+1, dtype=complex)
    pi_k = np.zeros(N//2+1, dtype=complex)
    
    # 处理直流分量（k=0，实数）
    phi_k[0] = np.random.normal(scale=np.sqrt(phi_trda2_k[0]))
    pi_k[0] = np.random.normal(scale=np.sqrt(pi_trda2_k[0]))
    
    # 处理Nyquist频率（当N为偶数时）
    if N % 2 == 0:
        phi_k[-1] = np.random.normal(scale=np.sqrt(phi_trda2_k[-1]))
        pi_k[-1] = np.random.normal(scale=np.sqrt(pi_trda2_k[-1]))
    
    # 处理其他正频率（复数分量）
    # 注意：长度N为偶数时，索引1到N//2（不包括Nyquist）
    # 长度N为奇数时，索引1到(N+1)//2-1
    start_idx = 1
    end_idx = (N//2) if N % 2 == 0 else (N+1)//2 - 1
    
    for i in range(start_idx, end_idx + 1):
        # 尺度参数：每个分量方差为功率谱值的一半
        scale_phi = np.sqrt(phi_trda2_k[i] / 2)
        re_phi = np.random.normal(scale=scale_phi)
        im_phi = np.random.normal(scale=scale_phi)
        phi_k[i] = re_phi + 1j * im_phi
        
        scale_pi = np.sqrt(pi_trda2_k[i] / 2)
        re_pi = np.random.normal(scale=scale_pi)
        im_pi = np.random.normal(scale=scale_pi)
        pi_k[i] = re_pi + 1j * im_pi
    
    # 逆变换到实空间
    phi_sample = np.fft.irfft(phi_k, n=N)
    pi_sample = np.fft.irfft(pi_k, n=N)
    
    return phi_sample, pi_sample

@numba.njit(fastmath=True, nogil = True)#W相空间中的每个点的W值
def W(phi, pi):
    kinetic = 0.5 * np.sum(pi**2)
    grad_x = finite_diff_gradient(phi)
    gradient = 0.5 * np.sum(grad_x**2) 
    potential = np.sum(V(phi))
    #W_val = 1 - beta * dx* (kinetic + gradient + potential)
    W_val = np.exp(-beta * dx* (kinetic + gradient + potential))
    return W_val

@numba.njit(fastmath=True, parallel=True, nogil = True)
def compute_W(phi_pairs, pi_pairs):
    print_boxed("Compute W Value", '#', 60)
    W_values = np.zeros((N_samples, N_samples))
    print("Sample\t\tTotal Samples\t\tTime Cost")
    # Compute W-values in parallel    
    for i in range(N_samples):
        for j in numba.prange(N_samples):
            W_values[i, j] = W(phi_pairs[i, j], pi_pairs[i, j])
        if i % 20 == 0:       
            print(f"{i}\t\t{N_samples}")            
    # Compute normalization factor once
    normalization_factor = np.sum(W_values) 
    W_values /= normalization_factor
    #print_boxed("Compute W Value Finished", '#', 60)
    return W_values    

@numba.njit(fastmath=True, parallel=False, nogil = True)
def kai(phi, pi, phi_remin, phi_remax):
    in_range = (phi >= phi_remin) & (phi < phi_remax)
    points_in_range = np.sum(in_range)
    ratio = points_in_range/N
    return ratio

@numba.njit(fastmath=True, parallel=False, nogil = True)
def compute_Kai(phi_pairs, pi_pairs, phi_remin, phi_remax):
    Kai = np.zeros((N_samples,N_samples))
    for i in range(N_samples):
        for j in range(N_samples):
            Kai[i,j] = kai(phi_pairs[i,j], pi_pairs[i,j], phi_remin, phi_remax)
    #print_boxed("Compute Kai Finished", '#', 60)
    return Kai

#@numba.njit(fastmath=True, parallel=False, nogil = True)
def compute_prob(W,K):
    numerator = np.sum(W*K)
    denominator = np.sum(W)
    prob = numerator/denominator
    return prob

def compute_gamma(Pfv_curr, Pr_curr, Pr_next, dt):
    # 防止除零错误
    if Pfv_curr == 0:
        return 0
    # 计算Gamma
    gamma = (Pr_next - Pr_curr) / (Pfv_curr * np.abs(dt))
    return gamma
    
# 主函数
if __name__ == '__main__':
    time_initial=time.time()
    parser = argparse.ArgumentParser()
    # 程序参数
    N_samples = 1000  # phi场和pi场的样本数
    dx = 0.25          # 空间步长
    dt = 0.01        # 时间步长 
    N_steps = 2    # 时间步数
    N_x = 1024 # 空间网格大小
    N = N_x
    length_x = N_x*dx # 无量纲的空间长度
    grid_shape = (N_x)

    parser.add_argument("--a", type=float, required=True, help="Quadratic coefficient")
    parser.add_argument("--b", type=float, required=True, help="Cubic coefficient")
    parser.add_argument("--c", type=float, required=True, help="Quartic coefficient")
    parser.add_argument("--beta", type=float, required=True,help="Inverse temperature parameter")
    args = parser.parse_args()

    #势能参数V=a*phi**2+b*phi**3+c*phi**4
    a = args.a
    b = args.b
    c = args.c 
 
    # 物理参数
    a_f = 1               # 共动坐标的尺度因子
    erfa = 0            # 控制度规的实参数
    wstar = 1     
    fstar = 1           # fstar、wstar为量纲为能量的标度因子
    R = 1.0
    lamda = 1
    #脚本命令行中读取beta
    beta = args.beta
    T = 1 / beta        # 无量纲温度 对应的物理温度Tphy=T*wstar
    
    #v = 0.22
    #m_eff = np.sqrt(lamda * (T**2 / 3 - v**2))

    #找出三个特殊点
    phi_l, phi_m1, phi_m, phi_m2, phi_n = find_special_points()
    print(f"phi_l: {phi_l}, phi_m1: {phi_m1}, phi_m: {phi_m}, phi_m2: {phi_m2}, phi_n: {phi_n}")
    V_max = V(phi_m)
    V_min = V(phi_m2)
    m_eff = V_2prime(phi_m1)**0.5  #有效质量
    print("m_eff=",m_eff)

    Lx = length_x/wstar  # 物理空间长度
    Dx = dx / wstar       # 有量纲的空间距离   dx无量纲
    Dt = dt * a_f**(erfa) / wstar   # 有量纲的时间距离    dt无量纲
    #坐标空间的离散动量 
    # 使用rfftfreq计算频率（仅非负频率）
    kx_rfft = np.fft.rfftfreq(N_x, d=dx) * 2 * np.pi
    k_squared = kx_rfft**2
    w_k = np.sqrt(k_squared / R**2 + m_eff**2)
    
    # 计算 |phi_trda(k)|^2 和 |pi_trda(k)|^2
    P_phi_k = 1 / w_k * 1 / (np.exp(w_k / T) - 1)   #这里的谱已经是模拟的了，P(物理)=P(模拟)/wstar
    P_pi_k = w_k * 1 / (np.exp(w_k / T) - 1)

    phi_trda2_k = (wstar / fstar) ** 2 * (N_x/ dx) * P_phi_k  #这里的dx和P_phi_k本身已经是模拟的量了，所以不需要再乘上wstar^2了
    pi_trda2_k = (wstar / fstar) ** 2 * (N_x / dx) * P_pi_k

    #save_field_in_kspace_spectrum(kx_rfft, phi_trda2_k)
    
    #划分场值区间
    N_intervals = 200
    phi_intervals = np.linspace(-0.5, 1.5, N_intervals + 1)
    
    plot_potential()
    #相关参数输出到一个参数文件
    parameter_file = "parameter.txt"
    write_param()

    print_boxed("Calculation Start", '%', 60)
    print_boxed("Generate Initial Sample", '#', 60)

    # 生成初始样本
    phi_samples = []
    pi_samples = []
    print("Sample\t\tTotal Samples\t\tTime Cost")
    last_time = time.time()
    for _ in range(N_samples):
        phi, pi = generate_initial_samples_corrected(phi_trda2_k, pi_trda2_k, N_x)
        phi_samples.append(phi)
        pi_samples.append(pi)
        if _ % 50 == 0:
            print(f"{_}\t\t{N_samples}\t\t\t{time.time() - last_time:.3f}s")
            last_time = time.time()
    phi_samples = np.array(phi_samples)
    pi_samples = np.array(pi_samples)

    # 计算理论功率谱对应的频率（使用rfftfreq）
    k_values = np.fft.rfftfreq(N_x, d=dx) * 2 * np.pi
    # 创建频谱对比图
    save_field_in_kspace_spectrum(phi_samples, phi_trda2_k, k_values, dx, "Phi")
    save_field_in_kspace_spectrum(pi_samples, pi_trda2_k, k_values, dx, "Pi")

    #创建保存样本对的数组
    phi_pairs = np.zeros((N_samples, N_samples, N_x), dtype=np.float64)
    pi_pairs = np.zeros((N_samples, N_samples, N_x), dtype=np.float64)

    # 将初始样本对赋值到样本对数组中
    for i in range(N_samples):
        for j in range(N_samples):
            phi_pairs[i, j] = phi_samples[i]  # i样本对应phi
            pi_pairs[i, j] = pi_samples[j]  # j样本对应pi
    print_boxed("Generate Initial Sample Finished", '*', 60)

    #绘制初始phi和pi值的直方图，查看样本的场值分布
    plot_fieldvalues(phi_pairs, "Phi")
    plot_fieldvalues(pi_pairs, "Pi")

    plot_save_fieldvalues(phi_pairs, "Phi", "phi_field_distribution.txt")
    plot_save_fieldvalues(pi_pairs, "Pi", "pi_field_distribution.txt")
    
    #查看某两个点处样本场值分布
    points = [(12), (10)] 
    save_field_point_distributions(phi_pairs, pi_pairs, points)
    #计算初始W
    Wigner0 = compute_W(phi_pairs, pi_pairs)
    print("W0(7,3)=", Wigner0[2,2])
    
    #创建平均场值分布文件
    open("avephidis.txt", "w").close()
    open("avepidis.txt", "w").close()
    
    # 生成5个随机数对，用来保存不同样本对场分布
    random_pairs = []
    while len(random_pairs) < 5:
        i = np.random.randint(1, N_samples + 1)
        j = np.random.randint(1, N_samples + 1)
        pair = (i, j)
        if pair not in random_pairs:
            random_pairs.append(pair)

    # 创建存放文件的目录和文件
    os.makedirs('./field_pairdis', exist_ok=True)
    for pair in random_pairs:
        phi_filename = f"./field_pairdis/phidis({pair[0]},{pair[1]}).txt"
        pi_filename  = f"./field_pairdis/pidis({pair[0]},{pair[1]}).txt"
        open(phi_filename, "w").close()  # 清空或创建文件
        open(pi_filename, "w").close()

    #创建保存图像的文件目录
    os.makedirs('./field_image', exist_ok=True)
    os.makedirs('./W_image', exist_ok=True)
    os.makedirs('./wave_function_distribution_image', exist_ok=True)
    #保存初始时刻的场
    plot_sample_average_field(phi_pairs, pi_pairs, 0)
    plot_sample_pair_field(phi_pairs, pi_pairs, 0)
    save_W_distribution_plots(Wigner0, 0)
    #save_Psi_squer(phi_pairs, pi_pairs, Wigner0, 0)    #初始时save_Psi_squer和save_Psi_squer_J算出来的是相同的

    #创建输出文件
    output_file = "gamma_values.txt"
    energe_file = "average_energe.txt" 
    with open(output_file, "w") as f:
        f.write("Step\t<phi>\t<pi>\tPfv\tPrv\tave_Kai_fv\tave_Kai_rv\tGamma\n") 
    with open(energe_file, "w") as f:
        f.write("Step\t<T>\t<Hi>\t<E>\n")
    
    P_fv_prev = None
    P_r_prev = None
    #save_steps = {5, 10, 167, 333, 500, 666, 833, 1000}  # 要保存波函数分布的几个时刻

    print_boxed("Start Iteration", '#', 60)

    for step in range(0, 1 + N_steps):
        last_time = time.time()
        print_boxed(f"Iteration {step} / {N_steps}", '#', 60) 
        phi_pairs, pi_pairs = substep2(phi_pairs, pi_pairs)         
        print("phi,pi updata finish")
        #Wigner=compute_W(phi_pairs, pi_pairs)
        print_boxed(f"Iteration {step} / {N_steps} Cost {time.time() - last_time:.3f}s", '%', 60)#打印出每次更新Jmp和场pairs花的时间
               
        # 每隔 50 步保存一次场分布图，并check平均能量是否守恒
        if step % 50 == 0 :    
               #plot_sample_average_field(phi_pairs, pi_pairs, step)
               #plot_sample_pair_field(phi_pairs, pi_pairs, step)
               #save_W_distribution_plots(Wigner, step)
               #save_Psi_squer(phi_pairs, pi_pairs, Wigner0, step)
               #save_average_field(phi_pairs, pi_pairs)
               #for pair in random_pairs:
                   #i, j = pair
                   #phi_data = phi_pairs[i-1, j-1]
                   #pi_data  = pi_pairs[i-1, j-1]
                   #save_pair_field(phi_data, pi_data, pair)
               #save_spectrum(phi_pairs, pi_pairs, step, dx, N_samples)
               print(f"Saved spectrum data for step {step}")
               phi_average = np.mean(phi_pairs)
               pi_average= np.mean(pi_pairs)
               ave_T, ave_Hi, ave_E = average_energy(phi_pairs, pi_pairs)
               with open(energe_file, "a") as f:
                   f.write(f"{step}\t{ave_T}\t{ave_Hi}\t{ave_E}\n")

        #if step % 100 == 0:
               #save_Psi_squer(phi_pairs, pi_pairs, Wigner0, step)

        #样本对统计因子
        Kai_t_fv = compute_Kai(phi_pairs, pi_pairs, -999, phi_m) #假真空区间的统计因子
        print(f"Kai_fv{step}(7,3)=", Kai_t_fv[7,3])
        Kf=np.sum(Kai_t_fv)/N_samples**2 #样本平均后的统计因子
        Kai_t_rv = compute_Kai(phi_pairs, pi_pairs, phi_m, 999) #真真空区间的统计因子
        print(f"Kai_rv{step}(7,3)=", Kai_t_rv[7,3])
        Kr=np.sum(Kai_t_rv)/N_samples**2

        P_fv = compute_prob(Wigner0, Kai_t_fv)
        print("P_fv=", P_fv)
        P_r = compute_prob(Wigner0, Kai_t_rv)
        print("P_r=", P_r)
        
        # 如果已经有前一步的 P_fv 和 P_r，计算Gamma
        if P_fv_prev is not None and P_r_prev is not None:
            Gamma = compute_gamma(P_fv_prev, P_r_prev, P_r, dt)
            print(f"Gamma at step {step} = {Gamma}")
            with open(output_file, "a") as f:
                f.write(f"{step}\t{phi_average}\t{pi_average}\t{P_fv}\t{P_r}\t{Kf}\t{Kr}\t{Gamma}\n")
        # 更新上一步的P_fv和P_r值
        P_fv_prev = P_fv
        P_r_prev = P_r

    print_boxed("Iteration Finished", '*', 60)
    time_final=time.time()
    all_time=time_final-time_initial

    print_boxed(f"All time cost:{all_time}", '!', 60)
    
    
    
    
    
    
