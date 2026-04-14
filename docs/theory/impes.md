# IMPES 方法

IMPES（Implicit Pressure, Explicit Saturation）是两相流模拟中最经典的方法。

## 核心思想

将两相流耦合方程拆分为：

1. **隐式求解压力方程** — 无条件稳定
2. **显式更新饱和度** — 有 CFL 条件限制

## 算法流程

```
┌─────────────────────────────────────────┐
│  1. 用当前 S_w 计算各相流度 λ_w, λ_o    │
│  2. 组装总流度压力方程: A·p = b          │
│  3. 隐式求解压力: p^{n+1} = A^{-1}·b    │
│  4. 用 p^{n+1} 计算各相达西流速          │
│  5. 用分流方程显式更新 S_w^{n+1}         │
│  6. 检查 CFL 条件                        │
└─────────────────────────────────────────┘
```

## 压力方程离散化

$$\frac{V_i \phi_i c_t}{\Delta t}(p_i^{n+1} - p_i^n) = \sum_j T_{ij}^t (p_j^{n+1} - p_i^{n+1}) + Q_{t,i}$$

其中总流度传导率：

$$T_{ij}^t = T_{ij} \cdot (\lambda_{w,i} + \lambda_{o,i})$$

## 饱和度显式更新

$$S_{w,i}^{n+1} = S_{w,i}^n + \frac{\Delta t}{V_i \phi_i} \left[\sum_j T_{ij}^w (p_j^{n+1} - p_i^{n+1}) + Q_{w,i}\right]$$

水相传导率：

$$T_{ij}^w = T_{ij} \cdot f_{w,upstream}$$

!!! warning "上游权重"
    分流函数必须使用**上游权重**（upstream weighting）：
    $$f_{w,up} = \begin{cases} f_w(S_{w,i}) & \text{if flow from } i \to j \\ f_w(S_{w,j}) & \text{if flow from } j \to i \end{cases}$$

## CFL 稳定性条件

显式饱和度更新要求时间步长满足 CFL 条件：

$$\Delta t \leq \frac{V_i \phi_i}{\left|\frac{\partial f_w}{\partial S_w}\right| \cdot |v_t| \cdot A_{face}}$$

实际应用中通常取安全系数 0.5-0.8。

## 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单 | CFL 条件限制时间步长 |
| 每步只需解一个线性系统 | 饱和度前沿可能数值弥散 |
| 内存占用小 | 不适合强非线性问题 |
| 计算效率高（小时间步时） | 大时间步时不如全隐式 |
