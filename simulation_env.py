# 文件名: simulation_env.py
import numpy as np
from collections import deque

class UserSimulator:
    def __init__(self, sensitivity=1.0, label="Standard User"):
        self.label = label
        self.sensitivity = sensitivity # 敏感度
        self.arousal = 0.0 # 用户自身的兴奋度 (隐藏状态)
        self.fs = 100.0
        self.time = 0.0
        
        # 物理结构耦合 (Simulation of the device vibrating the chassis)
        # 这是一个 FIR 滤波器，模拟电机震动传导到传感器
        self.structural_coupling = np.array([0.5, 0.3, 0.1]) 
        self.u_buffer = deque([0.0]*3, maxlen=3)

    def respond(self, u_input):
        self.time += 1/self.fs
        self.u_buffer.appendleft(u_input)
        
        # 1. 结构自耦合噪音 (这是 Ryder 必须自己滤除的)
        structural_noise = np.dot(self.structural_coupling, list(self.u_buffer))
        
        # 2. 用户生理反馈 (Biological Feedback)
        # 逻辑：如果输入的震动 u 强度适中且持续，用户的 Arousal 会上升
        # Arousal 上升会导致阴道肌肉微颤 (Micro-spasms)
        
        effective_stimulus = abs(u_input) * self.sensitivity
        self.arousal += effective_stimulus * 0.05
        self.arousal *= 0.999 # 自然消退
        
        # 产生肌肉压力波形
        # 呼吸波 (低频)
        breath = 0.1 * np.sin(2 * np.pi * 0.3 * self.time)
        
        # 肌肉共鸣 (与输入频率锁定，但有非线性)
        # 只有兴奋了才有反馈
        muscle_reaction = 0.0
        if self.arousal > 10.0:
            muscle_reaction = 0.5 * np.tanh(u_input * 2.0) * (self.arousal / 50.0)
            
        # 偶尔的身体移动 (Artifact)
        artifact = 0.0
        if np.random.random() < 0.005: # 0.5% 概率出现移动
            artifact = np.random.normal(0, 0.5)
            
        # 总压力观测值
        y_total = structural_noise + breath + muscle_reaction + artifact + np.random.normal(0, 0.01)
        
        return y_total