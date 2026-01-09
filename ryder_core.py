# 文件名: ryder_core.py
import numpy as np
from collections import deque

# --- 辅助函数 ---
def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ema(prev, x, alpha):
    return (1 - alpha) * prev + alpha * x

class RyderAgent:
    def __init__(self, fs=100.0):
        self.fs = fs
        self.dt = 1/fs
        
        # --- 1. 物理层：非牛顿流体属性 (Virtual Rheology) ---
        self.eta = 8.0          # 当前虚拟黏度 (Viscosity)
        self.eta_min = 0.5      # 极稀 (水态 - 高潮)
        self.eta_max = 12.0     # 极稠 (岩浆态 - 前戏)
        self.shear_memory = 0.0 # 剪切历史记忆 (触变性)
        
        # --- 2. 认知层：热力学状态 (Thermodynamics) ---
        self.heat = 0.0         # 唤起热量 (0-100)
        self.phase = "warmup"   # 当前阶段
        
        # --- 3. 感知层：自耦合消除 (LMS Filter) ---
        # 即使只有一个传感器，也能区分"由于我震动引起的波"和"用户肌肉的波"
        self.L = 5
        self.g_hat = np.zeros(self.L) # 估计的结构传递函数
        self.u_hist = deque([0.0]*self.L, maxlen=self.L)
        self.mu = 0.01          # 学习率
        
        # --- 4. 决策层：目标与输出 ---
        self.Y_star = 0.0       # 目标共鸣值 (Counterfactual Target)
        self.Y_real = 0.0       # 实际共鸣值
        self.freq = 0.5         # 输出频率
        self.amp = 0.0          # 输出幅度
        
        # --- 5. 数据层：CIEU 日志 (5元组) ---
        self.CIEU_log = []      # 存储 {X, U, Y_star, Y, R}
        
        # 缓存用于特征提取
        self.e_buf = deque([0.0]*int(fs), maxlen=int(fs)) 

    def _update_rheology(self, shear_force):
        """
        [非牛顿流体内核]
        剪切变稀：输入变化越剧烈(shear_force大)，黏度(eta)越低。
        这模拟了：激烈时刻 -> 阻尼变小 -> 穿透力变强。
        """
        # 触变性记忆：持续的剪切会让流体保持稀薄
        self.shear_memory = ema(self.shear_memory, shear_force, 0.05)
        
        # 物理公式：黏度随剪切率指数衰减
        target_eta = self.eta_max / (1.0 + 2.0 * (self.shear_memory ** 1.5))
        target_eta = clip(target_eta, self.eta_min, self.eta_max)
        
        # 缓慢变化，模拟流体的物理惯性
        self.eta = ema(self.eta, target_eta, 0.1)

    def _update_thermodynamics(self, resonance, artifact_level):
        """
        [热力学积分内核]
        共鸣(Resonance)积累热量，伪影(Artifact)导致冷却。
        """
        # 有效加热 = 共鸣强度 * (1 - 干扰)
        heating = 0.3 * resonance * (1.0 - artifact_level)
        # 自然冷却
        cooling = 0.05 + (0.2 * artifact_level) # 干扰越大，冷却越快(体验打断)
        
        delta = (heating - cooling) * (1.0 / self.fs) # 积分时间步
        self.heat = clip(self.heat + delta * 10.0, 0, 100) # *10为加速模拟因子
        
        # 阶段判断
        if self.heat < 30: self.phase = "warmup"
        elif self.heat < 80: self.phase = "climb"
        else: self.phase = "peak"

    def step(self, y_obs):
        """
        主循环：每 10ms 调用一次
        输入：y_obs (压力传感器读数)
        输出：u (电机电压/PWM)
        """
        # 1. LMS 自耦合消除：计算残差 e(t) = y(t) - g_hat * u(t-1)
        # e(t) 才是真正的"生物波形"
        u_vec = np.array(list(self.u_hist))
        y_pred = np.dot(self.g_hat, u_vec) if len(u_vec) == self.L else 0.0
        e = y_obs - y_pred
        
        # 在线学习结构传递函数 (LMS Update)
        if len(u_vec) == self.L:
            self.g_hat += self.mu * e * u_vec
        
        self.e_buf.append(e)
        self.u_hist.appendleft(self.amp) # 记录当前的输出作为下一次的历史
        
        # 2. 特征提取 (从残差 e 中)
        # 剪切力代理：残差的变化率
        shear = abs(e - self.e_buf[-2]) if len(self.e_buf) > 2 else 0.0
        # 伪影代理：巨大的基线漂移
        artifact = 1.0 if abs(e) > 0.8 else 0.0 
        
        # 3. 更新虚拟物理属性 (Rheology)
        self._update_rheology(shear)
        
        # 4. 更新热力学与目标 (Slow Loop Logic - 这里简化为每步更新)
        # Y_real (共鸣): 简单的能量重合度 (Correlation)
        resonance = clip(abs(e * self.amp) * 5.0, 0, 1) 
        self.Y_real = resonance
        
        self._update_thermodynamics(resonance, artifact)
        
        # 设定目标 Y_star (Sweet Spot)
        if self.phase == "warmup": self.Y_star = 0.3
        elif self.phase == "climb": self.Y_star = 0.6
        else: self.Y_star = 0.9
        
        # 5. 生成控制信号 U (Active Driving)
        # 核心逻辑：热量越高 -> 频率越高; 黏度越低 -> 幅度越大
        
        # 频率随热量爬升
        target_freq = 0.5 + (self.heat / 100.0) * 4.0 # 0.5Hz -> 4.5Hz
        self.freq = ema(self.freq, target_freq, 0.01)
        
        # 幅度受黏度控制 (Viscosity Damping)
        # 黏度大(eta高) -> 幅度被压制 (闷闷的感觉)
        # 黏度小(eta低) -> 幅度释放 (穿透的感觉)
        damping_factor = self.eta_min / self.eta # 0.0 ~ 1.0
        
        # 目标控制：Gap = Y_star - Y_real
        # 如果共鸣不足，尝试增加驱动力，但受限于黏度
        gap = self.Y_star - self.Y_real
        drive_force = 0.2 + 0.8 * sigmoid(gap * 2.0)
        
        target_amp = drive_force * damping_factor
        
        # 安全限幅与斜率限制 (Tungsten Motor Slew Rate)
        # 钨钢电机惯性大，不能突变，必须 ema 平滑
        self.amp = ema(self.amp, target_amp, 0.05)
        
        # 生成波形
        time_step = len(self.CIEU_log) * self.dt
        u_waveform = self.amp * np.sin(2 * np.pi * self.freq * time_step)
        
        # 6. 记录 CIEU 五元组
        reward = -abs(self.Y_star - self.Y_real) # 奖励 = 距离甜区的负误差
        
        log_entry = {
            "X": {"heat": self.heat, "eta": self.eta, "phase": self.phase},
            "U": {"amp": self.amp, "freq": self.freq, "raw": u_waveform},
            "Y_star": self.Y_star,
            "Y": self.Y_real,
            "R": reward
        }
        self.CIEU_log.append(log_entry)
        
        return u_waveform