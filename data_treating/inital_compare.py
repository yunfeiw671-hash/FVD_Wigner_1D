import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
import matplotlib.ticker as ticker
#对比一些参考量在不同条件下的不同情况

def parse_negative_wigner_stats(file_path):
    
    data = np.loadtxt(file_path, skiprows=1)
    count_negative = np.mean(data[0:4, 1])
    ratio = count_negative/1000**2
    return ratio

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
        
        # 验证必要参数是否存在
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

def parse_energy_file(file_path, volume):
    """
    解析average_energe.txt文件获取初始能量密度
    返回 rho_E = 第二行第三列数值 / volume
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # 检查文件完整性
            if len(lines) < 2:
                print(f"文件行数不足 @ {file_path}")
                return None
                
            # 处理第二行数据
            second_line = lines[1].strip()
            columns = second_line.split()
            
            if len(columns) < 2:
                print(f"第二行列数不足 @ {file_path}")
                return None
                
            try:
                energy = float(columns[3])
                return energy / volume  # 计算能量密度
            except ValueError:
                print(f"数值转换失败 @ {file_path}: '{columns[1]}'")
    
    except Exception as e:
        print(f"读取能量文件失败 @ {file_path}: {str(e)}")
    
    return None

def process_data(base_dir): #图例是Vb，画的不同Vb下rho_E/Vb随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb_value: {'betas': [], 'ratios': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
                # 提取beta值
                beta = float(beta_folder.split('=')[1])
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
                V_min = params['V_min']

                # ========== 解析能量文件 ==========
                energy_file = os.path.join(beta_path, 'average_energe.txt')
                if not os.path.exists(energy_file):
                    print(f"    ⚠️ 能量文件缺失")
                    continue
                
                rho_E = parse_energy_file(energy_file, volume)
                if rho_E is None:
                    print(f"    ⚠️ 能量数据无效")
                    continue
                
                # ========== 计算结果 ==========
                miu = 10
                lamda = miu/beta
                v = lamda**(1/4)

                ratio =  (Vb - V_min) / (rho_E-V_min) 
                
                # ========== 存储数据 ==========
                if Vb not in results:
                    results[Vb] = {'betas': [], 'ratios': []}
                
                results[Vb]['betas'].append(beta)
                results[Vb]['ratios'].append(ratio)
                print(f"    ✓ 成功记录: Vb={Vb:.4f}, ratio={ratio:.4e}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for Vb in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[Vb]['betas'])
        results[Vb]['betas'] = np.array(results[Vb]['betas'])[sorted_indices].tolist()
        results[Vb]['ratios'] = np.array(results[Vb]['ratios'])[sorted_indices].tolist()
    
    return results

def plot_results(results, output_dir):  
    """
    可视化结果
    """
    plt.figure(figsize=(8, 5))
    
    # 颜色配置（排除黄色）
    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        '#808000',   # 橄榄
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (Vb, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(data['betas'], data['ratios'],
                 marker=marker,
                 linestyle='-',
                 color=color,
                 markersize=8,
                 linewidth=2,
                 label=f'Vb = {Vb:.3f}')
    
    # 图表装饰
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(r'$V_b/\rho_E$', fontsize=14)
    plt.yscale('log')
    plt.title(r'$V_b/\rho_E$ vs $\beta$', fontsize=16)
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)   
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(1, color='r', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "rhoE_over_Vb_vs_beta.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()

def process_data2(base_dir):  #图例是Vb/dV=V_delat,画的不同Vb+dV下(rho_E+dV)/(Vb+dV)随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb-V_min: {'betas': [], 'ratios': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
                # 提取beta值
                beta = float(beta_folder.split('=')[1])
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
                V_min = params['V_min']
                #V_delta = Vb /(- V_min)
                V_delta = Vb
                #V_delta = Vb/(-V_min) #这里采用这种图例的原因是gamma与Vb是负相关的，跟1/ΔV也是负相关的

                # ========== 解析能量文件 ==========
                energy_file = os.path.join(beta_path, 'average_energe.txt')
                if not os.path.exists(energy_file):
                    print(f"    ⚠️ 能量文件缺失")
                    continue
                
                rho_E = parse_energy_file(energy_file, volume)
                if rho_E is None:
                    print(f"    ⚠️ 能量数据无效")
                    continue
                
                ratio =  (rho_E-V_min) / (Vb - V_min)
                #ratio =  rho_E / Vb 
                # ========== 存储数据 ==========
                if V_delta not in results:
                    results[V_delta] = {'betas': [], 'ratios': []}
                
                results[V_delta]['betas'].append(beta)
                results[V_delta]['ratios'].append(ratio)
                print(f"    ✓ 成功记录: V_delta={V_delta:.4f}, ratio={ratio:.4e}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for V_delta in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[V_delta]['betas'])
        results[V_delta]['betas'] = np.array(results[V_delta]['betas'])[sorted_indices].tolist()
        results[V_delta]['ratios'] = np.array(results[V_delta]['ratios'])[sorted_indices].tolist()
    
    return results

def plot_results2(results, output_dir):   
    """
    可视化结果
    """
    plt.figure(figsize=(18, 12))
    
    # 颜色配置（排除黄色）
    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        '#808000',   # 橄榄
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (V_delta, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(data['betas'], data['ratios'],
                 marker=marker,
                 linestyle='-',
                 color=color,
                 markersize=15,
                 linewidth=2,
                 label=r'$V_b$ 'f'= {V_delta:.3f}')

    # 图表装饰
    plt.xlabel(r'$\beta\omega^{\star}$', fontsize=35)
    plt.ylabel(r'$\rho_E/\bar{V}_b$', fontsize=35)
    #plt.ylabel(r'$\frac{\rho_E+\Delta V}{V_b+\Delta V}$', fontsize=35)

    # 增大坐标轴刻度字体
    # x_min = min(min(data['betas']) for data in results.values())  # 自动计算最小值
    # x_max = max(max(data['betas']) for data in results.values())  # 自动计算最大值
    x_min = 1.2
    x_max = 3.2
    tick_interval = 0.4  # 设置你想要的刻度间隔
    num_ticks = int((x_max - x_min)/tick_interval) + 1
    custom_ticks = np.linspace(x_min, x_max, num_ticks)
    plt.xticks(custom_ticks, [f"{x:.1f}" for x in custom_ticks], fontsize=35)  # 添加刻度字体设置
    
    ax = plt.gca()
    ax.set_yscale('log')
    def log_fmt(y, pos):
        if y == 1.0:
            return r'$10^{0}$'  # 特殊处理1.0
        else:
            return r'$10^{%.0f}$' % np.log10(y)  # 其他值保持科学计数
    
    # 只显示主刻度，不显示次刻度
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_fmt))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.axvline(x=2.18, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.31, color='purple', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.52, color='#3498db', linestyle='--', linewidth=1.5)
    plt.axvline(x=2.68, color='green', linestyle='--', linewidth=1.5)

    plt.yticks(fontsize=35)  # 添加刻度字体设置
    #plt.title(r'$\frac{\rho_E+\Delta V}{V_b+\Delta V}$ vs $\beta\omega^{\star}$', fontsize=16)
    plt.legend(fontsize=27, loc='upper right', frameon=True, shadow=True)   
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(1, color='brown', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "rhoE_over_Vb.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()

def process_data3(base_dir):  #图例是Vb/dV=V_delat,画的不同Vb+dV下（omega0/volume）/rho_E随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb-V_min: {'betas': [], 'ratios': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
                # 提取beta值
                beta = float(beta_folder.split('=')[1])
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
                V_min = params['V_min']
                V_delta = Vb /(- V_min)
                Vpara_a = params['Vpara_a']
                omega = (2*Vpara_a)**0.5
                rho_omega = omega/volume

                # ========== 解析能量文件 ==========
                energy_file = os.path.join(beta_path, 'average_energe.txt')
                if not os.path.exists(energy_file):
                    print(f"    ⚠️ 能量文件缺失")
                    continue
                
                rho_E = parse_energy_file(energy_file, volume)
                if rho_E is None:
                    print(f"    ⚠️ 能量数据无效")
                    continue
                
                # ========== 计算结果 ==========
                miu = 1
                lamda = miu/beta
                v = lamda**(1/4)

                ratio = rho_E / rho_omega   
                
                # ========== 存储数据 ==========
                if V_delta not in results:
                    results[V_delta] = {'betas': [], 'ratios': []}
                
                results[V_delta]['betas'].append(beta)
                results[V_delta]['ratios'].append(ratio)
                print(f"    ✓ 成功记录: V_delta={V_delta:.4f}, ratio={ratio:.4e}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for V_delta in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[V_delta]['betas'])
        results[V_delta]['betas'] = np.array(results[V_delta]['betas'])[sorted_indices].tolist()
        results[V_delta]['ratios'] = np.array(results[V_delta]['ratios'])[sorted_indices].tolist()
    
    return results

def plot_results3(results, output_dir):   
    plt.figure(figsize=(18, 10))
    
    # 颜色配置（排除黄色）
    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        '#808000',   # 橄榄
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (V_delta, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
    
        plt.plot(data['betas'], data['ratios'],
                 marker=marker,
                 linestyle='-',
                 color=color,
                 markersize=8,
                 linewidth=2,
                 label=r'$\frac{V_b}{\Delta V}$ 'f'= {V_delta:.3f}')

    # 图表装饰
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(r'$\rho_E/\rho_{\omega_0}$', fontsize=14)
    plt.yscale('log')
    plt.title(r'$\rho_E/\rho_{\omega_0}$ vs $\beta$', fontsize=16)
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)   
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(1, color='r', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "rhoE_over_omegarho_vs_beta.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()

def process_data4(base_dir):  #图例是Vb/dV=V_delat,画的不同Vb+dV下 beta*H 随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb-V_min: {'betas': [], 'ratios': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
                # 提取beta值
                beta = float(beta_folder.split('=')[1])
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
                V_min = params['V_min']
                V_delta = Vb /(- V_min)
                Vpara_a = params['Vpara_a']
                omega = (2*Vpara_a)**0.5
                rho_omega = omega/volume

                # ========== 解析能量文件 ==========
                energy_file = os.path.join(beta_path, 'average_energe.txt')
                if not os.path.exists(energy_file):
                    print(f"    ⚠️ 能量文件缺失")
                    continue
                
                rho_E = parse_energy_file(energy_file, volume)
                if rho_E is None:
                    print(f"    ⚠️ 能量数据无效")
                    continue
                

                ratio =  beta * rho_E * volume 
                
                # ========== 存储数据 ==========
                if V_delta not in results:
                    results[V_delta] = {'betas': [], 'ratios': []}
                
                results[V_delta]['betas'].append(beta)
                results[V_delta]['ratios'].append(ratio)
                print(f"    ✓ 成功记录: V_delta={V_delta:.4f}, ratio={ratio:.4e}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for V_delta in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[V_delta]['betas'])
        results[V_delta]['betas'] = np.array(results[V_delta]['betas'])[sorted_indices].tolist()
        results[V_delta]['ratios'] = np.array(results[V_delta]['ratios'])[sorted_indices].tolist()
    
    return results

def plot_results4(results, output_dir):   
    """
    可视化结果
    """
    plt.figure(figsize=(18, 10))
    
    # 颜色配置（排除黄色）
    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        '#808000',   # 橄榄
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (V_delta, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
    
        plt.plot(data['betas'], data['ratios'],
                 marker=marker,
                 linestyle='-',
                 color=color,
                 markersize=8,
                 linewidth=2,
                 label=r'$\frac{V_b}{\Delta V}$' f'= {V_delta:.3f}')

    # 图表装饰
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(r'$\beta * H$', fontsize=14)
    plt.yscale('log')
    #plt.title(r'$\beta * H $ vs $\beta$', fontsize=16)
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)   
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(1, color='r', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "beta multiply Hi.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()

def process_data5(base_dir):  #图例是Vb+dV=V_delat,画的不同Vb+dV下 “-W个数占比” 随beta的变化情况
    """
    主数据处理函数
    返回数据结构：{Vb-V_min: {'betas': [], 'ratios': []}}
    """
    results = {}
    
    # 第一层循环：遍历所有dV目录
    for dV_folder in os.listdir(base_dir):
        dV_path = os.path.join(base_dir, dV_folder)
        
        # 跳过非目标目录
        if not os.path.isdir(dV_path) or not dV_folder.startswith('dV='):
            continue
        
        print(f"\n处理目录: {dV_folder}")
        
        # 第二层循环：遍历当前dV下的所有beta目录
        for beta_folder in os.listdir(dV_path):
            beta_path = os.path.join(dV_path, beta_folder)
            
            if not os.path.isdir(beta_path) or not beta_folder.startswith('beta='):
                continue
            
            try:
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
                V_min = params['V_min']
                V_delta = Vb /(-V_min)

                # 提取beta值
                beta = float(beta_folder.split('=')[1])
                print(f"  ▸ 处理beta={beta:.3f}")

                Wigner_file = os.path.join(beta_path, 'negative_wigner_stats.txt')
                if not os.path.exists(Wigner_file):
                    print(f"    ⚠️ 参数文件缺失: {Wigner_file}")
                    continue
                
                ratio =  parse_negative_wigner_stats(Wigner_file)
                
                # ========== 存储数据 ==========
                if V_delta not in results:
                    results[V_delta] = {'betas': [], 'ratios': []}
                
                results[V_delta]['betas'].append(beta)
                results[V_delta]['ratios'].append(ratio)
                print(f"    ✓ 成功记录: V_delta={V_delta:.4f}, ratio={ratio:.4e}")
                
            except Exception as e:
                print(f"    处理异常: {str(e)}")
                continue
    
    # 对每个Vb的数据按beta排序
    for V_delta in results:
        # 将数据按beta排序
        sorted_indices = np.argsort(results[V_delta]['betas'])
        results[V_delta]['betas'] = np.array(results[V_delta]['betas'])[sorted_indices].tolist()
        results[V_delta]['ratios'] = np.array(results[V_delta]['ratios'])[sorted_indices].tolist()
    
    return results

def plot_results5(results, output_dir):   
    """
    可视化结果
    """
    plt.figure(figsize=(18, 10))
    
    # 颜色配置（排除黄色）
    colors = [
        '#2ecc71',   # 翠绿
        '#3498db',   # 亮蓝
        '#9b59b6',   # 紫色
        '#e74c3c',   # 红色
        '#1abc9c',   # 青绿
        '#00008B',   # 深蓝
        '#808000',   # 橄榄
        '#4B0082'    # 靛青
    ]
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # 绘制每条曲线
    for idx, (V_delta, data) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
    
        plt.plot(data['betas'], data['ratios'],
                 marker=marker,
                 linestyle='-',
                 color=color,
                 markersize=8,
                 linewidth=2,
                 label=r'$\frac{V_b}{\Delta V}$ 'f'= {V_delta:.3f}')

    # 图表装饰
    plt.xlabel(r'$\beta \omega^{\star}$', fontsize=14)
    plt.ylabel(r'$\frac{N_{W<0}}{N_W}$', fontsize=14)
    plt.yscale('log')
    plt.title(r'$\frac{N_{W<0}}{N_W}$ vs $\beta \omega^{\star}$', fontsize=16)
    plt.legend(fontsize=10, loc='lower right', frameon=True, shadow=True)   
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(1, color='r', linestyle='--')
    # 保存图片
    plot_path = os.path.join(output_dir, "negative_W_ratio.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {plot_path}")
    plt.show()


def plot_beta_distributions(root_dir, save_path=None):
    """
    自动绘制不同beta值的场值统计分布对比图
    :param root_dir: 包含所有beta=...文件夹的根目录（例如dV=0.06路径下不同beta=的文件夹）
    :param save_path: 图片保存路径
    """
    # 配置颜色循环（排除黄色）
    colors = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                             '#17becf', '#bcbd22'])
    plt.rc('axes', prop_cycle=colors)

    # 收集所有beta文件夹
    beta_folders = [f for f in os.listdir(root_dir) 
                   if f.startswith('beta=') and os.path.isdir(os.path.join(root_dir, f))]
    
    # 按beta值排序（假设文件夹命名格式为 beta=数值）
    beta_folders.sort(key=lambda x: float(x.split('=')[1]))

    # 准备绘图
    plt.figure(figsize=(20, 12), dpi=100)
    ax = plt.gca()

    # 遍历所有beta文件夹
    for folder in beta_folders:
        # 构建文件路径
        file_path = os.path.join(root_dir, folder, 'phi_field_distribution.txt')
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：{file_path} 不存在，已跳过")
            continue

        # 读取数据
        try:
            data = np.loadtxt(file_path)
            x = data[:, 0]
            y = data[:, 1]
        except Exception as e:
            print(f"读取 {file_path} 失败：{str(e)}")
            continue

        # 提取beta值（从文件夹名）
        beta_value = float(folder.split('=')[1].replace('_', '.'))  # 处理可能的_代替小数点

        # 绘制曲线
        ax.plot(x, y, lw=3, alpha=0.8, label=r'$(\beta\omega^{\star})$'f'={beta_value:.1f}')

    # 图形修饰
    ax.set_xlabel(r"$\phi/f^{\star}$", fontsize=35)
    ax.set_ylabel("Frequency", fontsize=35)
    # 增大坐标轴刻度字体
    plt.xticks(fontsize=35)  # 添加刻度字体设置
    plt.yticks(fontsize=35)  # 添加刻度字体设置
    #ax.set_title("Initial $\phi$ Distribution Comparison", fontsize=14)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    plt.axvline(0.4081542677427536, color='r', linestyle='--', label=r"$\phi_m$")

    # 智能图例布局
    handles, labels = ax.get_legend_handles_labels()
    ncol = 3 if len(labels) > 10 else (1 if len(labels) > 5 else 1)
    ax.legend(handles, labels, loc='upper left', ncol=ncol, fontsize=30)

    # 输出结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"对比图已保存至：{save_path}")
        plt.show()
    else:
        print(f"error")

if __name__ == "__main__":

    # 配置路径
    # base_directory = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\initial"
    # output_dir = base_directory  # 结果保存到原目录
    
    # 处理数据
    # results = process_data2(base_directory)
    
    # #保存原始数据
    # if results:
    #     data_file = os.path.join(output_dir, "rhoE_plus_dV_over_Vdelta_vs_beta.txt")
    #     #data_file = os.path.join(output_dir, "rhoE_over_omegarho_vs_beta.txt")
    #     #data_file = os.path.join(output_dir, "beta multiply Hi.txt")
    #     with open(data_file, 'w') as f:
    #         f.write("Vb+dV\tBeta\tRatio\n")
    #         for Vb, data in results.items():
    #             for beta, ratio in zip(data['betas'], data['ratios']):
    #                 f.write(f"{Vb:.6f}\t{beta}\t{ratio:.6e}\n")
    #     print(f"\n原始数据已保存至: {data_file}")
    
    # #绘制图表
    # plot_results2(results, output_dir)


    distribution_directory = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\initialdV=0.060"
    plot_beta_distributions(
        root_dir=distribution_directory,
        save_path=os.path.join(distribution_directory, "initial_phi_dis_comparison.pdf")
    )





