# 文件名: ryder_core.py
# --- 钨钢电机适配版 (强动力参数) ---

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
        
        # --- 1. 物理层：非牛顿流体属性 (已调优) ---
        self.eta = 6.0          # 当前虚拟黏度 (初始值)
        self.eta_min = 0.5      # 极稀 (水态 - 高潮)
        # [修改点1] 降低了最大粘度，从 12.0 改为 6.0，让它更容易动起来
        self.eta_max = 6.0      
        self.shear_memory = 0.0 # 剪切历史记忆
        
        # --- 2. 认知层：热力学状态 ---
        self.heat = 0.0         # 唤起热量 (0-100)
        self.phase = "warmup"   # 当前阶段
        
        # --- 3. 感知层：自耦合消除 ---
        self.L = 5
        self.g_hat = np.zeros(self.L) 
        self.u_hist = deque([0.0]*self.L, maxlen=self.L)
        self.mu = 0.01          
        
        # --- 4. 决策层：目标与输出 ---
        self.Y_star = 0.0       
        self.Y_real = 0.0       
        self.freq = 0.5         
        self.amp = 0.0          
        
        # --- 5. 数据层：CIEU 日志 ---
        self.CIEU_log = []      
        self.e_buf = deque([0.0]*int(fs), maxlen=int(fs)) 

    def _update_rheology(self, shear_force):
        self.shear_memory = ema(self.shear_memory, shear_force, 0.05)
        target_eta = self.eta_max / (1.0 + 2.0 * (self.shear_memory ** 1.5))
        target_eta = clip(target_eta, self.eta_min, self.eta_max)
        self.eta = ema(self.eta, target_eta, 0.1)

    def _update_thermodynamics(self, resonance, artifact_level):
        heating = 0.3 * resonance * (1.0 - artifact_level)
        cooling = 0.05 + (0.2 * artifact_level) 
        delta = (heating - cooling) * (1.0 / self.fs) 
        self.heat = clip(self.heat + delta * 20.0, 0, 100) # 加快了热量积累速度
        
        if self.heat < 30: self.phase = "warmup"
        elif self.heat < 80: self.phase = "climb"
        else: self.phase = "peak"

    def step(self, y_obs):
        u_vec = np.array(list(self.u_hist))
        y_pred = np.dot(self.g_hat, u_vec) if len(u_vec) == self.L else 0.0
        e = y_obs - y_pred
        
        if len(u_vec) == self.L:
            self.g_hat += self.mu * e * u_vec
        
        self.e_buf.append(e)
        self.u_hist.appendleft(self.amp) 
        
        shear = abs(e - self.e_buf[-2]) if len(self.e_buf) > 2 else 0.0
        artifact = 1.0 if abs(e) > 0.8 else 0.0 
        
        self._update_rheology(shear)
        
        resonance = clip(abs(e * self.amp) * 5.0, 0, 1) 
        self.Y_real = resonance
        self._update_thermodynamics(resonance, artifact)
        
        if self.phase == "warmup": self.Y_star = 0.3
        elif self.phase == "climb": self.Y_star = 0.6
        else: self.Y_star = 0.9
        
        target_freq = 0.5 + (self.heat / 100.0) * 3.0 
        self.freq = ema(self.freq, target_freq, 0.01)
        
        damping_factor = self.eta_min / self.eta 
        
        gap = self.Y_star - self.Y_real
        
        # [修改点2] 极大地增强了驱动力
        # 原来是 0.2 + 0.8... 现在是 2.0 + 3.0... 
        # 意思是：不管有没有共鸣，先给我震起来！
        drive_force = 2.0 + 3.0 * sigmoid(gap * 2.0)
        
        target_amp = drive_force * damping_factor
        
        self.amp = ema(self.amp, target_amp, 0.02)
        
        time_step = len(self.CIEU_log) * self.dt
        u_waveform = self.amp * np.sin(2 * np.pi * self.freq * time_step)
        
        reward = -abs(self.Y_star - self.Y_real) 
        
        log_entry = {
            "X": {"heat": self.heat, "eta": self.eta, "phase": self.phase},
            "U": {"amp": self.amp, "freq": self.freq, "raw": u_waveform},
            "Y_star": self.Y_star,
            "Y": self.Y_real,
            "R": reward
        }
        self.CIEU_log.append(log_entry)
        
        return u_waveform
