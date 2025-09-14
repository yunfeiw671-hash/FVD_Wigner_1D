import os
import shutil
import subprocess
import argparse
#使用时命令行输入：python run_batch.py --betas 0.6 0.8 1.0 1.2

# 预定义的[a,b,c,d]参数组合列表 V = a*phi^2 + b*phi^3 + c*phi^4, d为deltaV，即两个真空的势能差值
V_param_combinations = [
    (0.8, -1.8400290793253238, 0.9800221080139118, 0.06)   # V(trueVacuum)=-0.06, 极大值点 (m_x, m_y): (0.408154267742764, 0.0353580528465197)
    #(0.8, -1.9200306150809674, 1.0400221956507567, 0.08),   # V(trueVacuum)=-0.08, 极大值点 (m_x, m_y): (0.384606716234320, 0.0318603733703902)
    #(0.8, -2.0000281996123253, 1.1000224171152673, 0.1),   # V(trueVacuum)=-0.1, 极大值点 (m_x, m_y): (0.363629611530877, 0.0288495243900180)
    #(0.8, -2.080039056826153, 1.160029596985047, 0.12)     # V(trueVacuum)=-0.12, 极大值点 (m_x, m_y):(0.344818926366884, 0.0262401553974918)
    #(0.8, -2.3200061291458236, 1.3400034968360626, 0.18),   # V(trueVacuum)=-0.18, 极大值点 (m_x, m_y):(0.298506334387898, 0.0202151172773964)
    #(0.8, -2.5600011581525512, 1.5200025308051983, 0.24)   # V(trueVacuum)=-0.24, 极大值点 (m_x, m_y):(0.263157847130377, 0.0160373142558844)
    #(0.8, -2.800031134101623, 1.7000227944118098, 0.3)   # V(trueVacuum)=-0.3, 极大值点 (m_x, m_y): (0.235290862093400, 0.0130263282528754)
    #(0.8, -3.0399829062214487, 1.8799881679767496, 0.36)    # V(trueVacuum)=-0.36,极大值点 (m_x, m_y): (0.212767438607783, 0.0107877308156212)
    #(8.85, -19.699995376, 10.349996485, 0.5)
]

# 需要自动处理的输出文件和目录列表
output_items = [
    "field_image",
    "field_pairdis",
    "W_image",
    "wave_function_distribution_image",
    "spectrum",
    "avephidis.txt",
    "avepidis.txt",
    "average_energe.txt",
    "gamma_values.txt",
    "parameter.txt", 
    "phi_field_distribution.txt",
    "pi_field_distribution.txt",
    "Phi Value Distrubion.png",
    "Phi_spectrum_comparison.png",
    "Pi_spectrum_comparison.png",
    "Pi Value Distrubion.png",
    "potential_function.png"
]


def run_simulation(a, b, c, dV, beta_values):
    """双层遍历执行函数"""
    # 创建dV目录
    dV_dir = f"dV={dV:.3f}"  
    os.makedirs(dV_dir, exist_ok=True)

    for beta in beta_values:
        beta_dir = os.path.join(dV_dir, f"beta={beta:.3f}")
        os.makedirs(beta_dir, exist_ok=True)

        try:
            # 运行主程序（需支持参数传递）
            cmd = [
                "python", "FVD_wigner_1D.py",
                "--a", str(a),
                "--b", str(b),
                "--c", str(c),
                "--beta", str(beta)
            ]
            subprocess.check_call(cmd)

            # 移动输出文件
            for item in output_items:
                src = os.path.join(".", item)
                if os.path.exists(src):
                    dest = os.path.join(beta_dir, item)
                    
                    if os.path.exists(dest):
                        if os.path.isdir(dest):
                            shutil.rmtree(dest)
                        else:
                            os.remove(dest)
                    shutil.move(src, dest)

        except Exception as e:
            print(f"Error in dV={dV} beta={beta}: {str(e)}")
            # 清理残留文件
            for item in output_items:
                if os.path.exists(item):
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
            continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双层参数扫描")
    parser.add_argument("--betas", nargs="+", type=float, required=True,
                      help="要扫描的beta值列表，如 0.6 0.8 1.0")
    args = parser.parse_args()

    for a, b, c, dV in V_param_combinations:
        print(f"\n{'#'*40}")
        print(f"Processing dV={dV} with a={a}, b={b}, c={c}")
        print(f"{'#'*40}")
        
        run_simulation(a, b, c, dV, args.betas)



        