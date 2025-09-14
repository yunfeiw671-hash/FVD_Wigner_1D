import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
# 1. 读取数据
file_path = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\gamma_vs_beta.txt"
df = pd.read_csv(file_path, sep='\t')
out_path = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial"
plot_path = os.path.join(out_path, "e^-betaH_beta_gamma.pdf")
extra_file_path = r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\datagammaT_0.006.txt"  # instanton计算结果
df_extra = pd.read_csv(extra_file_path, sep='\t')
# 2. 取出数据列
Vb = df['Vb'].values
beta = df['Beta'].values
gamma = df['gamma'].values
log_gamma = np.log(gamma)

# 3. 定义二元函数模型，beta0 也是拟合参数
def log_gamma_model(X, a, b, c, beta0):
    beta, Vb = X
    return c + a * (beta - beta0) * np.sqrt(Vb) + b * ((beta - beta0)**2) * Vb

# 4. 构造输入
Xdata = np.vstack((beta, Vb))

# 5. 初始猜测参数 [a, b, c, beta0]
initial_guess = [1.0, -1.0, -1.0, np.mean(beta)]

# 6. 拟合
popt, pcov = curve_fit(log_gamma_model, Xdata, log_gamma, p0=initial_guess)
a, b, c, beta0_fitted = popt

# 7. 打印拟合参数
print("Fitted parameters:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"beta0 = {beta0_fitted}")

# 8. 绘图（gamma vs beta, 分不同 Vb）
plt.figure(figsize=(18, 12))
groups = df_extra['group'].unique()
extra_colors = ['black', 'darkred', 'darkgreen', 'orange', 'brown']
extra_markers = ['x', 'P', 'H', '8', '+']

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
# 添加额外的数据点到图中
for idx, grp in enumerate(groups):
    df_grp = df_extra[df_extra['group'] == grp]

    beta_extra = 1/df_grp['T'].values       # 横轴 βω*
    logGamma_over_T2 = -df_grp['-logΓ/T2'].values  # 纵轴 lnΓ/T²
    S_over_T = df_grp['S/T'].values
    gamma2 =  np.log(5*(S_over_T/(2*np.pi))**0.5*np.exp(-S_over_T))

    plt.plot(beta_extra, gamma2,
             linestyle='dashdot',
             color=colors[3-idx],
             #marker=extra_markers[idx % len(extra_markers)],
            #  label=f'Extra group {int(grp)}',
             linewidth=2)
    
unique_Vb = np.unique(Vb)
#colors = plt.cm.plasma(np.linspace(0, 1, len(unique_Vb)))

    
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

for i, Vb_fixed in enumerate(unique_Vb):
    mask = (Vb == Vb_fixed)
    beta_vals = beta[mask]
    gamma_vals = gamma[mask]
    marker = markers[i % len(markers)]

    # 拟合曲线
    beta_fit = np.linspace(min(beta_vals), max(beta_vals), 200)
    log_gamma_fit = log_gamma_model((beta_fit, np.full_like(beta_fit, Vb_fixed)), *popt)
    gamma_fit = log_gamma_fit

    # 画图
    plt.plot(beta_fit, gamma_fit, color=colors[i], label=r'$V_b$'f'={Vb_fixed:.3f} (fit)',linewidth=2)
    plt.scatter(beta_vals, np.log(gamma_vals),
                    marker=marker,
                    color=colors[i],
                    s=100,  # 点的大小
                    zorder=5,  # 使点在线上方
                    label=r'$V_b$ 'f'= {Vb_fixed:.3f}(sim)')
    #plt.scatter(beta_vals, np.log(gamma_vals), edgecolor='k', facecolor='none', s=40, label=f'Vb={Vb_fixed:.3f} (data)')

plt.xlabel(r'$\beta\omega^{\star}$', fontsize=35)
plt.ylabel(r'$ln(\Gamma/T)$', fontsize=35)
#plt.title(r'Fit of $\log(\gamma)$ vs $\beta$ and $V_b$', fontsize=16)
x_min = 1.2  # 自动计算最小值
x_max = 3.2  # 自动计算最大值
tick_interval = 0.4  # 设置你想要的刻度间隔
num_ticks = int((x_max - x_min)/tick_interval) + 1
custom_ticks = np.linspace(x_min, x_max, num_ticks)
plt.xticks(custom_ticks, [f"{x:.1f}" for x in custom_ticks], fontsize=35) 
plt.xlim(1.0, 3.4)
plt.ylim(-10,1)
plt.yticks(fontsize=35)
plt.axvline(x=2.18, color='red', linestyle='--', linewidth=1)
plt.axvline(x=2.31, color='purple', linestyle='--', linewidth=1)
plt.axvline(x=2.52, color='#3498db', linestyle='--', linewidth=1)
plt.axvline(x=2.68, color='green', linestyle='--', linewidth=1)
plt.legend(fontsize=25, loc='lower left', frameon=True, shadow=True)   
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.xscale('log')
plt.savefig(plot_path, dpi=300)
plt.show()

# 9. 计算拟合优度 R^2
log_gamma_pred = log_gamma_model((beta, Vb), *popt)
residuals = log_gamma - log_gamma_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_gamma - np.mean(log_gamma))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared = {r_squared:.6f}")