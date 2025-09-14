import numpy as np
import matplotlib.pyplot as plt
import os
import re
#用来画场演化图，取决于主程序运行是否保存了场值分布。

def plot_field_evolution(phi_filepath, pi_filepath, folder_path):
    # 提取文件名中的 pair
    match = re.search(r'phidis\((\d+),(\d+)\)\.txt', phi_filepath)
    if not match:
        print(f"Filename {phi_filepath} does not match expected pattern, skipping.")
        return
    i, j = match.groups()

    # 读取 phi 和 pi
    phi_data = np.loadtxt(os.path.join(folder_path, phi_filepath))
    pi_data  = np.loadtxt(os.path.join(folder_path, pi_filepath))

    # Plot phi
    plt.figure(figsize=(100, 18))
    plt.imshow(phi_data, aspect='auto', cmap='RdBu_r', origin='lower')
    plt.colorbar(label='Field Value (ϕ)')
    plt.xlabel('Field Point Index')
    plt.ylabel('Time Step')
    plt.title(f'Phi Field Evolution: Pair ({i},{j})')
    save_path = os.path.join(folder_path, f'phidis({i},{j}).png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

    # Plot pi
    plt.figure(figsize=(100, 18))
    plt.imshow(pi_data, aspect='auto', cmap='PuOr', origin='lower')
    plt.colorbar(label='Field Value (π)')
    plt.xlabel('Field Point Index')
    plt.ylabel('Time Step')
    plt.title(f'Pi Field Evolution: Pair ({i},{j})')
    save_path = os.path.join(folder_path, f'pidis({i},{j}).png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

def batch_process_all_fields(folder_path):
    # 扫描所有 phidis(*, *).txt 文件
    all_files = os.listdir(folder_path)
    phi_files = [f for f in all_files if re.match(r'phidis\(\d+,\d+\)\.txt', f)]

    print(f"Found {len(phi_files)} phi files.")

    for phi_file in phi_files:
        # 找到对应的 pi 文件
        match = re.search(r'phidis\((\d+),(\d+)\)\.txt', phi_file)
        if match:
            i, j = match.groups()
            pi_file = f'pidis({i},{j}).txt'
            if pi_file in all_files:
                plot_field_evolution(phi_file, pi_file, folder_path)
            else:
                print(f"Warning: Corresponding pi file {pi_file} not found for {phi_file}")

# 运行批处理
#folder_path=r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=2048_dx=0.25\dV=0.060\beta=1.800\field_pairdis"
folder_path=r"E:\Some_Documents\research\test_results\1D\test5_new_edition\meff=1.6\W_dx\Ns=1000\fix_meff\redefine_RFV\Wigner=e^-betaH\Nx=1024_dx=0.25\positive_dt\dt=0.01\fix_initial\dV=0.06\beta=1.200\field_pairdis"

batch_process_all_fields(folder_path)
