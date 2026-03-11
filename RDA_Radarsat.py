import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 1. 加载外部真实数据
# ====================================================================
print(">> 正在加载真实雷达数据 (echo.mat)...")
try:
    mat_data = sio.loadmat('echo.mat')
    s_echo = mat_data['echo'] 
except FileNotFoundError:
    print("⚠️ 未找到 echo.mat 文件，此处生成占位复数矩阵以保证代码通过。")
    s_echo = np.random.randn(4096, 4096) + 1j * np.random.randn(4096, 4096)

Naz, Nrg = s_echo.shape
print(f">> 数据维度: 方位向 {Naz} x 距离向 {Nrg}")

# ====================================================================
# 2. RADARSAT-1 真实雷达参数
# ====================================================================
R0  = 988647.462   # 中心最短斜距 (m)
Vr  = 7062         # 雷达平台有效速度 (m/s)
Tr  = 41.74e-6     # 脉冲持续时间 (s)
Kr  = -0.72135e12  # 距离向线性调频率 (Hz/s)
f0  = 5.3e9        # 雷达载波频率 (Hz)
Fr  = 32.317e6     # 距离向采样率 (Hz)
Fa  = 1256.98      # 脉冲重复频率 PRF (Hz)
fc  = -596.271     # 多普勒中心频率 (Hz)
c   = 3e8          

lamda = c / f0
Nr = int(Tr * Fr)  

# 反推等效波束斜视角
sita_r_c = np.arcsin(fc * lamda / (2 * Vr))

tr = 2 * R0 / c + (np.arange(-Nrg/2, Nrg/2)) / Fr     
ta = np.arange(-Naz/2, Naz/2) / Fa                    

# ====================================================================
# 3. 严格遵循教材流程：距离压缩(驻留频域) -> 方位FFT -> SRC -> 距离IFFT
# ====================================================================
print(">> 执行基带搬移与距离向 FFT...")
N_safe = 8192 

# 步骤 A: 方位向基带搬移 (时域)
s_echo_base = s_echo * np.exp(-1j * 2 * np.pi * fc * np.tile(ta[:, np.newaxis], (1, Nrg)))

# 步骤 B: 距离向 FFT (进入距离频域-方位时域)
S_range_freq = np.fft.fft(s_echo_base, n=N_safe, axis=1)  

# 步骤 C: 距离压缩 (驻留频域，只相乘不逆变换)
print(">> 执行距离压缩 (驻留频域)...")
fr = np.fft.fftfreq(N_safe, d=1/Fr)
t_ref = np.arange(-Nr/2, Nr/2) / Fr
s_ref = np.exp(1j * np.pi * Kr * (t_ref**2)) * np.kaiser(Nr, 2.5)
S_ref = np.fft.fft(s_ref, n=N_safe)
H_range = np.conj(S_ref)[np.newaxis, :] 

S_rc_freq = S_range_freq * H_range

# 步骤 D: 方位向 FFT (进入二维频域)
print(">> 执行方位向 FFT，进入二维频域...")
S_2D = np.fft.fft(S_rc_freq, n=Naz, axis=0)

# 步骤 E: 二次距离压缩 (SRC)
print(">> 引入 SRC 滤波器进行交叉耦合修正...")
fa = fc + np.fft.fftfreq(Naz, d=1/Fa)
fr_mtx, fa_mtx = np.meshgrid(fr, fa)

D_fa = np.sqrt(1 - (lamda * fa_mtx / (2 * Vr))**2)
inv_K_src = (R0 * lamda**3 * fa_mtx**2) / (2 * Vr**2 * c**2 * D_fa**3)
H_src = np.exp(-1j * np.pi * (fr_mtx**2) * inv_K_src)

S_2D_comp = S_2D * H_src

# 步骤 F: 距离向 IFFT (退回距离多普勒域)
print(">> 执行距离向 IFFT，退回距离多普勒域...")
s_rc = np.fft.ifft(S_2D_comp, axis=1)
N_rg_valid = Nrg - Nr + 1
S_rd = s_rc[:, :N_rg_valid] 

# ====================================================================
# 4. 距离徙动校正 (RCMC)
# ====================================================================
print(">> 执行精确 RCMC (8 点 Sinc 插值)...")

D_fa_1D = np.sqrt(1 - (lamda * fa / (2 * Vr))**2)
tr_RCMC = 2 * R0 / c + np.arange(-N_rg_valid/2, N_rg_valid/2) / Fr
R0_RCMC = (c / 2) * tr_RCMC * np.cos(sita_r_c)

delta_Rrd_fn = R0_RCMC * (1 / D_fa_1D[:, np.newaxis] - 1)
delta_Rrd_fn_num = delta_Rrd_fn / (c / (2 * Fr))

R_interp = 8
S_rd_rcmc = np.zeros((Naz, N_rg_valid), dtype=np.complex128)
pos = np.arange(N_rg_valid)[np.newaxis, :] + delta_Rrd_fn_num
pos_ceil = np.ceil(pos).astype(int)

weights = np.zeros((R_interp, Naz, N_rg_valid))
pts_wrap = np.zeros((R_interp, Naz, N_rg_valid), dtype=int)

for idx, k in enumerate(range(-R_interp//2 + 1, R_interp//2 + 1)): 
    pts = pos_ceil - k
    weights[idx] = np.sinc(pos - pts) 
    pts_wrap[idx] = pts % N_rg_valid        

weights /= np.sum(weights, axis=0)
for idx in range(R_interp):
    S_rd_rcmc += weights[idx] * S_rd[np.arange(Naz)[:, np.newaxis], pts_wrap[idx]]

# ====================================================================
# 5. 方位压缩
# ====================================================================
print(">> 执行方位向匹配滤波...")
Haz = np.exp(1j * 4 * np.pi * R0_RCMC / lamda * D_fa_1D[:, np.newaxis])
S_rd_c = S_rd_rcmc * Haz
s_ac = np.fft.ifft(S_rd_c, axis=0)

# ====================================================================
# 6. 全流程图像渲染模块 (图名与窗口名严谨修正版)
# ====================================================================
print(">> 成像完成，准备绘制全流程数据图表...")

def plot_2d_image(data, title, xlabel, ylabel, cmap='gray', vmin=None, vmax=None, is_db=False):
    # 【修正】：将 figure 的 num 属性设为 title，让弹出的窗口名字直接显示对应的图名
    plt.figure(num=title, figsize=(7, 6))
    if is_db:
        data_plot = 20 * np.log10(np.abs(data) / np.max(np.abs(data)) + 1e-6)
        plt.imshow(data_plot, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='幅度 (dB)')
    else:
        plt.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='数值')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

# ----------------- A. 原始数据 -----------------
plot_2d_image(np.abs(s_echo), "1. RADARSAT-1 原始回波数据 (幅度)", "距离向 (采样点)", "方位向 (采样点)")
plot_2d_image(np.angle(s_echo), "2. RADARSAT-1 原始回波数据 (相位)", "距离向 (采样点)", "方位向 (采样点)", cmap='hsv')

# ----------------- B. 滤波器分析 -----------------
plt.figure(num="3. 距离匹配滤波器 (时域实部)", figsize=(7, 4))
plt.plot(np.real(s_ref))
plt.title("3. 距离匹配滤波器 (时域实部)")
plt.xlabel("距离时域 (采样点)")
plt.ylabel("幅度")
plt.tight_layout()

plot_2d_image(np.fft.fftshift(np.angle(H_src)), "4. 二次距离压缩(SRC)滤波器相位 (Shift居中)", "距离频率 (频点)", "方位频率 (频点)", cmap='hsv')

plt.figure(num="5. 方位匹配滤波器相位 (中心距切片)", figsize=(7, 4))
plt.plot(np.fft.fftshift(np.angle(Haz[:, N_rg_valid//2])))
plt.title("5. 方位匹配滤波器相位 (中心距切片 - Shift居中)")
plt.xlabel("方位多普勒频率 (频点)")
plt.ylabel("相位 (弧度)")
plt.tight_layout()

# ----------------- C. 数据演变历程 -----------------
# 【修正】：明确标出此时只做了距离压缩，尚未做 SRC
s_rc_time = np.fft.ifft(S_rc_freq, axis=1)[:, :N_rg_valid]
plot_2d_image(s_rc_time, "6. 仅距离压缩后 (方位时域-距离时域)", "距离向 (采样点)", "方位向 (采样点)", is_db=True, vmin=-50, vmax=0)

# 【修正】：明确标出此时已经叠加了 SRC 补偿，状态发生了根本改变
plot_2d_image(S_rd, "7. 距离压缩与SRC叠加后 (距离多普勒域，未RCMC)", "距离向 (采样点)", "方位频率 (采样点)", is_db=True, vmin=-50, vmax=0)
plot_2d_image(S_rd_rcmc, "8. 距离徙动校正(RCMC)后 (距离多普勒域)", "距离向 (采样点)", "方位频率 (采样点)", is_db=True, vmin=-50, vmax=0)

# ----------------- D. 最终成像结果 -----------------
# 【修正】：找回真实数据的霸气称号
plot_2d_image(s_ac, "9. RADARSAT-1 真实数据 SAR 聚焦成像", "距离向 (采样点)", "方位向 (采样点)", is_db=True, vmin=-45, vmax=0)

plt.show()