# =========================================================================
# 大斜视角 RDA 点目标仿真
# =========================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 1. 定义参数
# ====================================================================
R_nc = 20e3                 # 景中心斜距
Vr = 150                    # 雷达有效速度
Tr = 2.5e-6                 # 发射脉冲时宽
Kr = 20e12                  # 距离调频率
f0 = 5.3e9                  # 雷达工作频率
BW_dop = 80                 # 多普勒带宽
Fr = 60e6                   # 距离采样率
Fa = 200                    # 方位采样率
Naz = 1024                  # 方位向采样点数
Nrg = 1024                  # 距离向采样点数
sita_r_c = (10 * np.pi) / 180  
c = 3e8                     

R0 = R_nc * np.cos(sita_r_c)    
Nr = int(Tr * Fr)               
BW_range = Kr * Tr              
lamda = c / f0                  
fnc = 2 * Vr * np.sin(sita_r_c) / lamda  
La_real = 0.886 * 2 * Vr * np.cos(sita_r_c) / BW_dop  
beta_bw = 0.886 * lamda / La_real        
La = 0.886 * R_nc * lamda / La_real      

NFFT_r = Nrg                
NFFT_a = Naz                

# ====================================================================
# 2. 设定仿真点目标的位置
# ====================================================================
delta_R0 = 0        
delta_R1 = 120      
delta_R2 = 50

x1 = R0 
y1 = delta_R0 + x1 * np.tan(sita_r_c)
x2 = x1 
y2 = y1 + delta_R1 
x3 = x2 + delta_R2 
y3 = y2 + delta_R2 * np.tan(sita_r_c) 

x_range = np.array([x1, x2, x3])
y_azimuth = np.array([y1, y2, y3])
nc_target = (y_azimuth - x_range * np.tan(sita_r_c)) / Vr

# ====================================================================
# 3. 时间与频率网格定义
# ====================================================================
tr = 2 * x1 / c + (np.arange(-Nrg/2, Nrg/2)) / Fr     
ta = np.arange(-Naz/2, Naz/2) / Fa                    
tr_mtx, ta_mtx = np.meshgrid(tr, ta)                  

# ====================================================================
# 4. 生成点目标原始数据
# ====================================================================
print(">> 开始生成回波数据...")
s_echo = np.zeros((Naz, Nrg), dtype=np.complex128)
A0 = 1 

for k in range(3):
    R_n = np.sqrt(x_range[k]**2 + (Vr * ta_mtx - y_azimuth[k])**2)  
    w_range = np.abs(tr_mtx - 2 * R_n / c) <= (Tr / 2)
    sita = np.arctan(Vr * (ta_mtx - nc_target[k]) / x_range[k])
    w_azimuth_sinc = (np.sinc(0.886 * sita / beta_bw))**2
    w_azimuth_rect = np.abs(ta_mtx - nc_target[k]) <= (La / 2) / Vr
    w_azimuth = w_azimuth_sinc * w_azimuth_rect
    s_k = A0 * w_range * w_azimuth * np.exp(-1j * 4 * np.pi * f0 * R_n / c) * np.exp(1j * np.pi * Kr * (tr_mtx - 2 * R_n / c)**2)
    s_echo += s_k

# ====================================================================
# 5 & 6. 距离压缩(驻留频域) -> 方位FFT -> SRC -> 距离IFFT
# ====================================================================
print(">> 执行基带搬移与距离向 FFT...")
N_safe = 2048

# 步骤 A: 方位向基带搬移 (时域)
s_echo_base = s_echo * np.exp(-1j * 2 * np.pi * fnc * np.tile(ta[:, np.newaxis], (1, Nrg)))

# 步骤 B: 距离向 FFT (进入距离频域-方位时域)
S_range_freq = np.fft.fft(s_echo_base, n=N_safe, axis=1)

# 步骤 C: 距离压缩 (只乘滤波器，不进行 IFFT，保留在距离频域)
print(">> 执行距离压缩 (驻留频域)...")
fr = np.fft.fftfreq(N_safe, d=1/Fr)
t_ref = np.arange(-Nr/2, Nr/2) / Fr
s_ref = np.exp(1j * np.pi * Kr * (t_ref**2)) * np.kaiser(Nr, 2.5)
S_ref = np.fft.fft(s_ref, n=N_safe)
H_range = np.conj(S_ref)[np.newaxis, :] 

S_rc_freq = S_range_freq * H_range # 此时完成了教材上的“不进行傅里叶逆变换的距离压缩”

# 步骤 D: 方位向 FFT (此时正式进入二维频域)
print(">> 执行方位向 FFT，进入二维频域...")
S_2D = np.fft.fft(S_rc_freq, n=NFFT_a, axis=0)

# 步骤 E: 二次距离压缩 (SRC)
print(">> 引入 SRC 滤波器进行交叉耦合修正...")
fa = fnc + np.fft.fftfreq(NFFT_a, d=1/Fa)
fr_mtx, fa_mtx = np.meshgrid(fr, fa)

D_fa = np.sqrt(1 - (lamda * fa_mtx / (2 * Vr))**2)
inv_K_src = (R0 * lamda**3 * fa_mtx**2) / (2 * Vr**2 * c**2 * D_fa**3)
H_src = np.exp(-1j * np.pi * (fr_mtx**2) * inv_K_src)

S_2D_comp = S_2D * H_src # 完成教材上的“二次距离压缩”步骤

# 步骤 F: 距离向 IFFT (退回到距离多普勒域，准备 RCMC)
print(">> 执行距离向 IFFT，退回距离多普勒域...")
s_rc = np.fft.ifft(S_2D_comp, axis=1)
N_rg = Nrg - Nr + 1
S_rd = s_rc[:, :N_rg]

# ====================================================================
# 7. 距离徙动校正 (RCMC)
# ====================================================================
print(">> 开始精确 RCMC (基于双曲线解析方程)...")

D_fa_1D = np.sqrt(1 - (lamda * fa / (2 * Vr))**2)
tr_RCMC = 2 * x1 / c + np.arange(-N_rg/2, N_rg/2) / Fr
R0_RCMC = (c / 2) * tr_RCMC * np.cos(sita_r_c)

delta_Rrd_fn = R0_RCMC * (1 / D_fa_1D[:, np.newaxis] - 1)

num_range = c / (2 * Fr)
delta_Rrd_fn_num = delta_Rrd_fn / num_range

R = 8
S_rd_rcmc = np.zeros((NFFT_a, N_rg), dtype=np.complex128)
pos = np.arange(N_rg)[np.newaxis, :] + delta_Rrd_fn_num
pos_ceil = np.ceil(pos).astype(int)

weights = np.zeros((R, NFFT_a, N_rg))
pts_wrap = np.zeros((R, NFFT_a, N_rg), dtype=int)

for idx, k in enumerate(range(-R//2 + 1, R//2 + 1)): 
    pts = pos_ceil - k
    weights[idx] = np.sinc(pos - pts) 
    pts_wrap[idx] = pts % N_rg        

weights /= np.sum(weights, axis=0)
for idx in range(R):
    S_rd_rcmc += weights[idx] * S_rd[np.arange(NFFT_a)[:, np.newaxis], pts_wrap[idx]]

# ====================================================================
# 8. 方位压缩
# ====================================================================
print(">> 开始基于解析相位的方位向压缩...")

Haz = np.exp(1j * 4 * np.pi * R0_RCMC / lamda * D_fa_1D[:, np.newaxis])
S_rd_c = S_rd_rcmc * Haz
s_ac = np.fft.ifft(S_rd_c, axis=0)


# ====================================================================
# 9. 统一绘图函数
# ====================================================================
print(">> 渲染图像中...")
def plot_img(data, title, idx):
    plt.figure(idx, figsize=(6, 5))
    plt.imshow(np.abs(data), aspect='auto', cmap='gray')
    plt.title(title)
    plt.xlabel('距离向 (采样点)')
    plt.ylabel('方位向 (采样点)')
    plt.colorbar()
    plt.tight_layout()

plot_img(s_echo, '图1：原始雷达回波数据', 1)
plot_img(S_rd,   '图2：距离多普勒域 (未 RCMC)', 2)
plot_img(S_rd_rcmc, '图3：距离多普勒域 (已进行矢量化 Sinc RCMC)', 3)
plot_img(s_ac,   '图4：最终 SAR 聚焦成像', 4)


# ====================================================================
# 10. 小斜视点目标质量分析 (IRW, PSLR, ISLR)
# ====================================================================
print(">> 开始对 3 个点目标进行量化指标评估...")
from scipy.signal import find_peaks

def analyze_1d_target(slice_1d, pixel_spacing, up_factor=16):
    N = len(slice_1d)
    F = np.fft.fftshift(np.fft.fft(np.fft.fftshift(slice_1d)))
    pad_len = N * up_factor
    F_pad = np.pad(F, (pad_len//2 - N//2, pad_len//2 - (N - N//2)), 'constant')
    slice_up = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(F_pad)))) * up_factor
    
    res_up = pixel_spacing / up_factor
    
    peak_idx = np.argmax(slice_up)
    peak_val = slice_up[peak_idx]
    slice_norm = slice_up / peak_val
    slice_db = 20 * np.log10(slice_norm + 1e-12)
    
    crossings = np.where(np.diff(np.sign(slice_db + 3)))[0]
    left_crosses = crossings[crossings < peak_idx]
    right_crosses = crossings[crossings >= peak_idx]
    
    if len(left_crosses) == 0 or len(right_crosses) == 0:
        return 0.0, 0.0, 0.0, slice_db, res_up, peak_idx
        
    left_idx, right_idx = left_crosses[-1], right_crosses[0]
    
    x_left = left_idx + (-3 - slice_db[left_idx]) / (slice_db[left_idx+1] - slice_db[left_idx])
    x_right = right_idx + (-3 - slice_db[right_idx]) / (slice_db[right_idx+1] - slice_db[right_idx])
    irw = (x_right - x_left) * res_up
    
    null_width = int((x_right - x_left) * 1.5)
    peaks, _ = find_peaks(slice_norm)
    sidelobes = [p for p in peaks if p < peak_idx - null_width or p > peak_idx + null_width]
    pslr = np.max(slice_db[sidelobes]) if len(sidelobes) > 0 else 0
    
    null_left = max(0, peak_idx - null_width)
    null_right = min(len(slice_norm)-1, peak_idx + null_width)
    E_main = np.sum(slice_norm[null_left:null_right]**2)
    E_total = np.sum(slice_norm**2)
    E_side = E_total - E_main
    islr = 10 * np.log10(E_side / E_main) if E_side > 0 else 0
    
    return irw, pslr, islr, slice_db, res_up, peak_idx

print("\n" + "="*55)
print(" 大斜视点目标成像质量评估报告 ")
print("="*55)

dx = c / (2 * Fr) 
dy = Vr / Fa       

SAR_abs = np.abs(s_ac)
temp_img = SAR_abs.copy()
max_y, max_x = SAR_abs.shape  

WIN = 16 

for target_id in range(3):
    center_y, center_x = np.unravel_index(np.argmax(temp_img), temp_img.shape)
    
    slice_azimuth = SAR_abs[max(0, center_y-WIN):min(max_y, center_y+WIN), center_x]
    slice_range = SAR_abs[center_y, max(0, center_x-WIN):min(max_x, center_x+WIN)]
    
    irw_r, pslr_r, islr_r, db_r, res_r, p_r = analyze_1d_target(slice_range, dx)
    irw_a, pslr_a, islr_a, db_a, res_a, p_a = analyze_1d_target(slice_azimuth, dy)
    
    print(f"▶ 目标 {target_id + 1} (位于矩阵坐标 Y:{center_y}, X:{center_x})")
    print(f"  [距离向 Range]   IRW: {irw_r:.3f} m | PSLR: {pslr_r:.2f} dB | ISLR: {islr_r:.2f} dB")
    print(f"  [方位向 Azimuth] IRW: {irw_a:.3f} m | PSLR: {pslr_a:.2f} dB | ISLR: {islr_a:.2f} dB")
    print("-" * 55)
    
    mask = 8
    temp_img[max(0, center_y-mask):min(max_y, center_y+mask), max(0, center_x-mask):min(max_x, center_x+mask)] = 0

print("✅ 所有目标测试完毕！")
print("="*55 + "\n")
plt.show()
print(">> 运行结束！")