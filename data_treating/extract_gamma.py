import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
#用于从输出文件中拟合提取不同条件下的衰变率，并保存到文件中

def extract_gamma(file_path, R_thresh, window_frac, step_in, step_out):   
    # 1. 加载数据，跳过第一行标题
    data = np.loadtxt(file_path, skiprows=1, usecols=(0, 3, 4, 7))  # 加载:Step、P_FV、P_RV、gamma
    T = dt*data[step_in:step_out, 0]  # 时间列
    P_FV = data[step_in:step_out, 1]
    P_RV = data[step_in:step_out, 2]
    RV_gamma = data[step_in:step_out, 3]

    # 2. 预处理
    valid_mask = (P_FV > 0)
    T_clean = T[valid_mask]
    log_PFV = np.log(P_FV[valid_mask])

    # 3. 滑动窗口拟合
    n = len(T_clean)
    window = max(int(n * window_frac), 10)
    valid_windows = []
    r_squared_list = []
    gamma_fit_list = []
    window_mid_T = []

    for i in range(n - window):
        t_segment = T_clean[i:i+window]
        y_segment = log_PFV[i:i+window]
        slope, _, r_value, _, _ = linregress(t_segment, y_segment)  #slope是拟合斜率，r_value用来衡量t和y之间线性相关程度，0 < r_value^2 < 1，越接近于1表示线性拟合程度越好
        gamma_fit = -slope   #负号是因为P = P0*exp(-gamma*T)
        r_squared = r_value**2
        r_squared_list.append(r_squared)
        gamma_fit_list.append(gamma_fit)
        window_mid_T.append(t_segment[window//2])  # 记录窗口中点时间   q

        if r_squared > R_thresh:   #利用拟合程度阈值来筛选
            valid_windows.append((i, i+window, gamma_fit, r_squared))  # 记录符合条件窗口
  
    # 4. 没有符合的情况
    if not valid_windows:
        return {"error": "No valid exponential decay region found."}

    # # 5. 利用 gamma 稳定性挑选gamma
    # gamma_stable = []
    # for start, end, _ ,_ in valid_windows:
    #     gamma_segment = RV_gamma[start:end]
    #     gamma_mean = np.mean(gamma_segment)
    #     gamma_std = np.std(gamma_segment)   #标准差，即这个区间内gamma值和平均gamma值的涨落有多大，越小表示相对越稳定
    #     gamma_stable.append((start, end, gamma_mean, gamma_std))

    # gamma_stable.sort(key=lambda x: x[3])  # 按 std 排序，把std依次从小到大排序

    # best_start, best_end, best_gamma, best_r2 = gamma_stable[0]
    # best_step_range = (T_clean[best_start], T_clean[best_end])

    # gamma_fit_R2: 满足R_thresh的窗口中R²最大的gamma
    max_r2 = max(w[3] for w in valid_windows)
    gamma_fit_R2 = next(w[2] for w in valid_windows if w[3] == max_r2)

    # 5. 找 gamma 最大的区间挑选gamma
    valid_windows.sort(key=lambda x: x[2], reverse=True)  # 按 x[2]则是gamma_fit 降序，x[3]则是best_r2降序
    best_start, best_end, best_gamma, best_r2 = valid_windows[0]
    best_step_range = (best_start+step_in, best_end+step_in)

    # start_indices = [w[0] for w in valid_windows]
    # end_indices = [w[1] for w in valid_windows]
    # best_start = min(start_indices)
    # best_end = max(end_indices)
    # best_step_range = (best_start+step_in, best_end+step_in)
    # gamma_values = [w[2] for w in valid_windows]
    # best_gamma = np.mean(gamma_values)

    #best_time_range = (T_clean[best_start], T_clean[best_end])

    RV_gamma_segment = RV_gamma[best_start:best_end]
    #ave_RV_gamma = np.mean(RV_gamma_segment)
    #ave_FV_gamma = np.mean(best_gamma)  #FV——Γ的平均值
    max_RV_gamma = max(RV_gamma_segment)
    #max_RV_gamma = max(RV_gamma)
    median_RV_gamma = np.median(RV_gamma_segment)
    #FV_gamma = max(gamma_fit)
    # 7. 返回
    return {
        "gamma": best_gamma,    #满足条件窗口中最大的FVgamma
        "gamma_fit_R2": gamma_fit_R2,  # 所有满足条件窗口中R²最大的gamma
        #"r_squared": best_r2,
        "step_range": best_step_range,
        "RV_gamma": max_RV_gamma,  #满足条件窗口中的RVgamma
        "unit": "beta^{-1}"
    }

def parse_parameter_file(file_path):
    """
    解析parameter.txt文件获取Vb和length_x
    返回字典格式：{'Vb': float, 'length_x': float}
    """
    params = {'Vb': None, 'length_x': None, 'V_min': None}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 提取势垒高度 Vb
                if line.startswith('(phi_m,V_max)'):
                    try:
                        # 示例行格式：(phi_m,V_max) = (0.408, 0.035)
                        value_part = line.split('=')[1].split('#')[0].strip()
                        values = value_part.strip('()').split(',')
                        if len(values) >= 2:
                            params['Vb'] = float(values[1].strip())
                    except Exception as e:
                        print(f"解析Vb失败 @ {file_path}: {str(e)}")
                
                # 提取空间长度 length_x
                elif line.startswith('length_x'):
                    try:
                        value = line.split('=')[1].split('#')[0].strip()
                        params['length_x'] = float(value)
                    except Exception as e:
                        print(f"解析length_x失败 @ {file_path}: {str(e)}")
                # 提取真真空势能 V_min
                elif line.startswith('(phi_m2,V_min)'):
                    try:
                        value_part = line.split('=')[1].split('#')[0].strip()
                        values = value_part.strip('()').split(',')
                        if len(values) >= 2:
                            params['V_min'] = float(values[1].strip())
                    except Exception as e:
                        print(f"解析V_min失败 @ {file_path}: {str(e)}")
                # 提取势能参数a
                elif line.startswith('a ='):
                    try:
                        value = line.split('=')[1].split('#')[0].strip()
                        params['Vpara_a'] = float(value)
                    except Exception as e:
                        print(f"解析势能参数a失败 @ {file_path}: {str(e)}")
        
        if params['Vb'] is None:
            print(f"⚠️ 未找到Vb参数 @ {file_path}")
        if params['length_x'] is None:
            print(f"⚠️ 未找到length_x参数 @ {file_path}")
        if params['V_min'] is None:
            print(f"⚠️ 未找到V_min参数 @ {file_path}")
        if params['Vpara_a'] is None:
            print(f"⚠️ 未找到Vpara_a参数 @ {file_path}")
            
    except Exception as e:
        print(f"读取文件失败 @ {file_path}: {str(e)}")
    
    return params

def process_data(base_dir):  #图例是Vb/dV=V_delat,画的不同Vb+dV下gamma随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb-V_min: {'betas': [], 'gamma': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        dV = float(dV_folder.split('=')[1])
        if dV == 0.06:
            betac = 2.18
        elif dV == 0.10:
            betac =  2.31
        elif dV == 0.18:
            betac = 2.52
        elif dV == 0.24:
            betac = 2.68
 
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
                # 提取beta值
                beta = float(beta_folder.split('=')[1])
                Tem = 1 / beta
                print(f"  ▸ 处理beta={beta:.3f}")
                
                # ========== 解析参数文件 ==========
                param_file = os.path.join(beta_path, 'parameter.txt')
                if not os.path.exists(param_file):
                    print(f"    ⚠️ 参数文件缺失: {param_file}")
                    continue
                
                params = parse_parameter_file(param_file)
                
                # 验证参数完整性
                if params['Vb'] is None or params['length_x'] is None or params['V_min'] is None:
                    print(f"    ⚠️ 参数不完整")
                    continue
                
                Vb = params['Vb']
                volume = params['length_x']  # 1D体积即为长度
                #volume = 1
                V_min = params['V_min']
                V_delta = Vb /(- V_min)

                 # 处理gamma数据
                gamma_file = os.path.join(beta_path, 'gamma_values.txt')
                if not os.path.exists(gamma_file):
                    print(f"Gamma file missing in {beta_path}")
                    continue

                if beta < betac:
                    result = extract_gamma(gamma_file, 0.99, 0.25, 0, 1000)
                    gamma_T = result['gamma'] / Tem
                    step_range = result['step_range']
                elif beta >= betac :
                    result = extract_gamma(gamma_file, 0.99, 0.5, 1000, 4000)
                    gamma_T = result['gamma'] / Tem
                    step_range = result['step_range']
                # elif beta == 3.4:
                #     result = extract_gamma(gamma_file, 0.7, 0.2, 1999, 4000)
                #     gamma_T = result['gamma'] / Tem
                #     step_range = result['step_range']
                
                if "error" in result:
                    print(f"Error in {beta_folder}: {result['error']}")
                    continue

                # ========== 存储数据 ==========
                if V_delta not in results:
                    results[V_delta] = {'Vb': [], 'betas': [], 'gamma_T': [], 'step_range': []}
                
                results[V_delta]['Vb'].append(Vb)
                results[V_delta]['betas'].append(beta)    #如果横轴要改成温度，直接把这里的beta替换成1/beta
                results[V_delta]['gamma_T'].append(gamma_T)
                results[V_delta]['step_range'].append(step_range)
                print(f"    ✓ 成功记录: V_delta={V_delta:.4f}, gamma={gamma_T:.4e}, step_range={step_range}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for V_delta in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[V_delta]['betas'])
        results[V_delta]['Vb'] = np.array(results[V_delta]['Vb'])[sorted_indices].tolist()
        results[V_delta]['betas'] = np.array(results[V_delta]['betas'])[sorted_indices].tolist()
        results[V_delta]['gamma_T'] = np.array(results[V_delta]['gamma_T'])[sorted_indices].tolist()
        results[V_delta]['step_range'] = np.array(results[V_delta]['step_range'])[sorted_indices].tolist()
    
    return results
def plot_results(results, output_dir):   #图例是Vb/dV=V_delat,画的不同Vb+dV下(Vb+dV)/(rho_E+dV)随beta的变化情况
    plt.figure(figsize=(18, 12))

    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        "#741A52",   # 
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (V_delta, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        gamma_Ts = data['gamma_T']
        loggamma_Ts = np.log(gamma_Ts)
        plt.scatter(data['betas'], gamma_Ts,
                    marker=marker,
                    color=color,
                    s=100,  # 点的大小
                    zorder=5,  # 使点在线上方
                    label=r'$\frac{V_b}{\Delta V}$ 'f'= {V_delta:.3f}')
    
    # 图表装饰
    plt.xlabel(r'$\beta\omega^{\star}$', fontsize=35)
    plt.ylabel(r'$\Gamma/T$', fontsize=35)
    #plt.ylabel(r'$\frac{\Gamma}{T L}\frac{1}{\omega^{\star}}$', fontsize=14)
    x_min = min(min(data['betas']) for data in results.values())  # 自动计算最小值
    x_max = max(max(data['betas']) for data in results.values())  # 自动计算最大值
    tick_interval = 0.4  # 设置你想要的刻度间隔
    num_ticks = int((x_max - x_min)/tick_interval) + 1
    custom_ticks = np.linspace(x_min, x_max, num_ticks)
    plt.xticks(custom_ticks, [f"{x:.1f}" for x in custom_ticks], fontsize=35) 
    # 增大坐标轴刻度字体
    plt.xticks(fontsize=35)  # 添加刻度字体设置
    plt.yticks(fontsize=35)  # 添加刻度字体设置
    plt.axvline(x=2.18, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.31, color='purple', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.52, color='#3498db', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.68, color='green', linestyle='--', linewidth=1.5)

    plt.yscale('log')
    #plt.title(r'$\frac{\Gamma}{TL}\frac{1}{\omega^{\star}}$ vs $\beta\omega^{\star} $ ', fontsize=16)
    #plt.title(r'$log(\frac{\Gamma}{TL}\frac{1}{\omega^{\star}})$ vs $\beta\omega^{\star} $ ', fontsize=16)
    #plt.title(r'$log(\frac{\Gamma}{TL}\frac{1}{\omega^{\star}})$ vs $\beta\omega^{\star} —— Contain W(\phi,\Pi)$ ', fontsize=16)
    #plt.title(r'$log(\frac{\Gamma}{TL}\frac{1}{\omega^{\star}})$ vs $\beta\omega^{\star} —— Just ~ \chi(\phi,\Pi)$ ', fontsize=16)
    plt.legend(fontsize=30, loc='lower left', frameon=True, shadow=True)    
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    #plt.axhline(1, color='r', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "e^-betaH_beta_gamma.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()
if __name__ == "__main__":
    # 配置路径
    #base_directory = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt"
    base_directory = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial"
    output_dir = base_directory  # 结果保存到原目录
    dt = 0.01
    # 处理数据
    results = process_data(base_directory)
    
    # 保存原始数据
    if results:
        #data_file = os.path.join(output_dir, "rhoE_plus_dV_over_Vdelta_vs_beta.txt")
        data_file = os.path.join(output_dir, "gamma_vs_beta.txt")
        with open(data_file, 'w') as f:
            f.write("Vb/dV\tVb\tBeta\tgamma\tstep_range\n")
            for V_delta, data in results.items():
                for Vb, beta, gamma_T, step_range in zip(data['Vb'], data['betas'], data['gamma_T'], data['step_range']):
                    f.write(f"{V_delta:.6f}\t{Vb:.6f}\t{beta}\t{gamma_T:.6e}\t{step_range}\n")
        print(f"\n原始数据已保存至: {data_file}")

    # 绘制图表
    plot_results(results, output_dir)