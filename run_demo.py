# 文件名: run_demo.py
import numpy as np
import matplotlib.pyplot as plt
from ryder_core import RyderAgent
from simulation_env import UserSimulator

# 1. 初始化
# 模拟一个慢热型用户
user = UserSimulator(sensitivity=0.8, label="User B (Slow Warmup)")
# 初始化 Ryder 智能体
ryder = RyderAgent()

# 2. 运行模拟 (20分钟 = 1200秒)
# 为了演示方便，我们加速模拟，跑 3000 个时间步 (约 30秒的高密度交互)
steps = 3000
time_axis = np.arange(steps) * 0.01 # 10ms per step

history_heat = []
history_eta = []
history_u = []
history_y = []
history_ystar = []
history_yreal = []

print("System Initialized. Starting Session...")

for t in range(steps):
    # --- 物理闭环 ---
    
    # 1. 上一时刻的观测 (假设初始为0)
    y_obs = history_y[-1] if len(history_y) > 0 else 0.0
    
    # 2. Ryder 思考并输出动作 U
    u_out = ryder.step(y_obs)
    
    # 3. 作用于用户，得到新的观测 Y
    y_new = user.respond(u_out)
    
    # --- 记录数据 ---
    history_u.append(u_out)
    history_y.append(y_new)
    
    # 从 log 中提取内部状态
    last_log = ryder.CIEU_log[-1]
    history_heat.append(last_log["X"]["heat"])
    history_eta.append(last_log["X"]["eta"])
    history_ystar.append(last_log["Y_star"])
    history_yreal.append(last_log["Y"])

print("Session Complete. Generating CIEU Report...")

# 3. 绘图分析
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 图1: 热力学与流变 (X - Context)
ax1.plot(time_axis, history_heat, 'r-', linewidth=2, label='Heat (Arousal)')
ax1.set_ylabel('Heat (0-100)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1_twin = ax1.twinx()
ax1_twin.plot(time_axis, history_eta, 'b--', label='Viscosity (Eta)')
ax1_twin.set_ylabel('Viscosity (Thickness)', color='b')
ax1_twin.tick_params(axis='y', labelcolor='b')
ax1.set_title(f'Ryder Context X: Thermodynamics & Non-Newtonian State ({user.label})')
ax1.grid(True, alpha=0.3)

# 图2: 目标与现实 (Y* vs Y)
ax2.plot(time_axis, history_ystar, 'g--', linewidth=2, label='Target Y* (Sweet Spot)')
ax2.plot(time_axis, history_yreal, 'k-', alpha=0.6, label='Actual Y (Resonance)')
ax2.fill_between(time_axis, history_ystar, history_yreal, color='gray', alpha=0.2, label='Gap (Reward Loss)')
ax2.set_ylabel('Resonance Level')
ax2.set_title('Causal Loop: Target Y* vs Real Y')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 干预输出 (U - Intervention)
ax3.plot(time_axis, history_u, 'm-', linewidth=1, label='Motor Output U')
ax3.set_ylabel('Voltage/PWM')
ax3.set_xlabel('Time (s)')
ax3.set_title('Intervention U: Adaptive Waveform')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()