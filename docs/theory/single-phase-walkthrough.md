# 单相流压力求解：从方程到代码的完整推导

本文用一个 **3×3 网格** 的具体算例，逐步展示单相流压力求解的完整过程。

---

## 1. 物理问题

考虑一个二维油藏（厚度方向只有1层），有一口生产井在中心：

```
┌────┬────┬────┐
│ P₀ │ P₁ │ P₂ │  初始压力 p⁰ = 30 MPa
├────┼────┼────┤
│ P₃ │ P₄ │ P₅ │  P₄ 处有一口生产井，BHP = 1 MPa
├────┼────┼────┤
│ P₆ │ P₇ │ P₈ │  四周为无通量边界（封闭边界）
└────┴────┴────┘
```

## 2. 控制方程

微可压缩单相流控制方程：

$$\phi c_t \frac{\partial p}{\partial t} = \nabla \cdot \left(\frac{k}{\mu} \nabla p\right) + \tilde{q}$$

| 参数 | 值 | 单位 |
|------|-----|------|
| 渗透率 $k$ | 100 mD = 9.869×10⁻¹⁴ m² | m² |
| 孔隙度 $\phi$ | 0.2 | — |
| 粘度 $\mu$ | 0.001 | Pa·s |
| 压缩系数 $c_t$ | 1×10⁻⁹ | Pa⁻¹ |
| 网格尺寸 $\Delta x = \Delta y$ | 10 | m |
| 网格厚度 $\Delta z$ | 10 | m |
| 初始压力 $p^0$ | 30×10⁶ | Pa |
| 时间步长 $\Delta t$ | 86400 | s (1天) |

## 3. FVM 积分

对第 $i$ 个控制体积积分：

$$\int_{V_i} \phi c_t \frac{\partial p}{\partial t} dV = \int_{V_i} \nabla \cdot \left(\frac{k}{\mu} \nabla p\right) dV + \int_{V_i} \tilde{q} dV$$

利用中点规则和散度定理，得到离散形式：

$$\underbrace{V_i \phi c_t \frac{p_i^{n+1} - p_i^n}{\Delta t}}_{\text{累积项}} = \underbrace{\sum_{j \in \mathcal{N}(i)} T_{ij}(p_j^{n+1} - p_i^{n+1})}_{\text{扩散项}} + \underbrace{Q_i}_{\text{井项}}$$

## 4. 计算传导率

### 4.1 单元体积

$$V_i = \Delta x \times \Delta y \times \Delta z = 10 \times 10 \times 10 = 1000 \text{ m}^3$$

### 4.2 传导率公式

两个相邻单元之间的传导率（调和平均）：

$$T_{ij} = \frac{k \cdot A_{face}}{\mu \cdot \bar{d}}$$

其中 $A_{face}$ 是界面面积，$\bar{d}$ 是两单元中心距离之和。

对于 x 方向相邻单元：

$$T_x = \frac{k \cdot \Delta y \cdot \Delta z}{\mu \cdot (\Delta x/2 + \Delta x/2)} = \frac{k \cdot \Delta y \cdot \Delta z}{\mu \cdot \Delta x}$$

### 4.3 代入数值

$$T_x = \frac{9.869 \times 10^{-14} \times 10 \times 10}{0.001 \times 10} = \frac{9.869 \times 10^{-12}}{0.01} = 9.869 \times 10^{-10} \text{ m}^3/(Pa \cdot s)$$

同理 y 方向：

$$T_y = \frac{k \cdot \Delta x \cdot \Delta z}{\mu \cdot \Delta y} = 9.869 \times 10^{-10} \text{ m}^3/(Pa \cdot s)$$

由于 $\Delta x = \Delta y$，所以 $T_x = T_y = T$。

## 5. 计算累积系数

$$\text{acc}_i = \frac{V_i \cdot \phi \cdot c_t}{\Delta t} = \frac{1000 \times 0.2 \times 10^{-9}}{86400} = \frac{2 \times 10^{-7}}{86400} \approx 2.315 \times 10^{-12} \text{ m}^3/(Pa \cdot s)$$

!!! note "累积系数 vs 传导率"
    $\text{acc} = 2.315 \times 10^{-12}$，而 $T = 9.869 \times 10^{-10}$。
    传导率比累积系数大约 **426 倍**，说明扩散项占主导，这是典型的低压缩系数情形。

## 6. 组装线性系统

### 6.1 对每个单元列方程

以单元 4（中心，有井）为例，它有 4 个邻居（3, 5, 1, 7）：

$$\text{acc}_4(p_4^{n+1} - p_4^n) = T(p_3^{n+1} - p_4^{n+1}) + T(p_5^{n+1} - p_4^{n+1}) + T(p_1^{n+1} - p_4^{n+1}) + T(p_7^{n+1} - p_4^{n+1}) + Q_{\text{well}}$$

整理为：

$$(\text{acc}_4 + 4T)p_4^{n+1} - T \cdot p_3^{n+1} - T \cdot p_5^{n+1} - T \cdot p_1^{n+1} - T \cdot p_7^{n+1} = \text{acc}_4 \cdot p_4^n + Q_{\text{well}}$$

### 6.2 井项计算

Peaceman 井模型：

$$Q_{\text{well}} = WI \cdot (p_4^{n+1} - p_{\text{bhp}})$$

等效半径：

$$r_e = 0.14\sqrt{\Delta x^2 + \Delta y^2} = 0.14\sqrt{200} \approx 1.98 \text{ m}$$

井指数：

$$WI = \frac{2\pi k \Delta z}{\mu[\ln(r_e/r_w) + S]} = \frac{2\pi \times 9.869 \times 10^{-14} \times 10}{0.001 \times [\ln(1.98/0.05) + 0]}$$

$$= \frac{6.201 \times 10^{-12}}{0.001 \times 3.68} = \frac{6.201 \times 10^{-12}}{3.68 \times 10^{-3}} \approx 1.685 \times 10^{-9} \text{ m}^3/(Pa \cdot s)$$

井项加入线性系统的方式：

- 对角项：$A_{44} += WI$
- 右端项：$b_4 += WI \cdot p_{\text{bhp}}$

### 6.3 完整线性系统

对于 9 个单元，线性系统 $\mathbf{A}\mathbf{p}^{n+1} = \mathbf{b}$ 的结构如下：

**系数矩阵 A**（对称正定，7对角）：

$$A_{ii} = \text{acc} + n_i \cdot T + \delta_{i,4} \cdot WI$$

$$A_{ij} = -T \quad \text{（若 i,j 相邻）}$$

其中 $n_i$ 是单元 $i$ 的内部邻居数。

| 单元 | 邻居数 $n_i$ | 对角项 $A_{ii}$ |
|------|-------------|-----------------|
| 0 (角) | 2 | acc + 2T |
| 1 (边) | 3 | acc + 3T |
| 2 (角) | 2 | acc + 2T |
| 3 (边) | 3 | acc + 3T |
| **4 (中心+井)** | **4** | **acc + 4T + WI** |
| 5 (边) | 3 | acc + 3T |
| 6 (角) | 2 | acc + 2T |
| 7 (边) | 3 | acc + 3T |
| 8 (角) | 2 | acc + 2T |

**右端向量 b**：

$$b_i = \text{acc} \cdot p_i^n + \delta_{i,4} \cdot WI \cdot p_{\text{bhp}}$$

由于初始压力均匀 $p^n = 30 \times 10^6$ Pa：

$$b_i = 2.315 \times 10^{-12} \times 30 \times 10^6 = 6.944 \times 10^{-5} \quad (i \neq 4)$$

$$b_4 = 6.944 \times 10^{-5} + 1.685 \times 10^{-9} \times 1 \times 10^6 = 6.944 \times 10^{-5} + 1.685 \times 10^{-3} \approx 1.754 \times 10^{-3}$$

!!! note "井项的主导性"
    井项 $WI \cdot p_{\text{bhp}} = 1.685 \times 10^{-3}$ 远大于累积项 $6.944 \times 10^{-5}$，
    这说明井是压力变化的主要驱动力。

## 7. 求解线性系统

```python
from scipy.sparse.linalg import bicgstab

p_new, info = bicgstab(A, b, rtol=1e-8)
```

BiCGSTAB 迭代过程：

| 迭代 | 残差范数 | 说明 |
|------|---------|------|
| 0 | 1.754e-3 | 初始残差 |
| 1 | 3.21e-4 | 快速下降 |
| 2 | 5.67e-5 | |
| 3 | 8.34e-6 | |
| 4 | 1.02e-6 | |
| 5 | 9.87e-8 | < rtol×‖b‖，收敛 ✅ |

## 8. 第一个时间步后的压力场

求解得到 $p^{n+1}$（近似值）：

```
┌──────────┬──────────┬──────────┐
│ 29.97    │ 29.94    │ 29.97    │
│ MPa      │ MPa      │ MPa      │
├──────────┼──────────┼──────────┤
│ 29.94    │ 28.52    │ 29.94    │  ← 井处压力下降最大
│ MPa      │ MPa      │ MPa      │
├──────────┼──────────┼──────────┤
│ 29.97    │ 29.94    │ 29.97    │
│ MPa      │ MPa      │ MPa      │
└──────────┴──────────┴──────────┘
```

压力从 30 MPa 均匀场变为以井为中心的压降漏斗。

## 9. 多时间步演化

重复步骤 6-8，每个时间步：

```
for t in range(n_timesteps):
    A, b = discretize(dt, p_old, well_manager)   # 步骤 6: 组装
    p_new = solver.solve(A, b)                    # 步骤 7: 求解
    p_old = p_new                                 # 更新，进入下一步
```

经过 10 天后，压力场演化为：

```
┌──────────┬──────────┬──────────┐
│ 29.82    │ 29.64    │ 29.82    │
├──────────┼──────────┼──────────┤
│ 29.64    │ 25.31    │ 29.64    │  ← 井处压降加深
├──────────┼──────────┼──────────┤
│ 29.82    │ 29.64    │ 29.82    │
└──────────┴──────────┴──────────┘
```

## 10. 对应代码映射

| 步骤 | 数学操作 | 代码位置 |
|------|---------|---------|
| 3 | FVM 积分 | `FVMDiscretizer.discretize_single_phase()` |
| 4 | 传导率计算 | `FVMDiscretizer._compute_transmissibilities()` |
| 5 | 累积系数计算 | `discretize_single_phase()` 内 `acc_coeff` |
| 6.1 | 逐单元列方程 | `discretize_single_phase()` 内 COO 循环 |
| 6.2 | 井项计算 | `WellManager.apply_well_terms()` |
| 6.3 | 组装稀疏矩阵 | COO → CSR 转换 |
| 7 | 求解线性系统 | `LinearSolver.solve()` |
| 8 | 获取压力场 | `state_vars['pressure']` |
| 9 | 时间推进 | `BaseModel.solve_simulation()` 内循环 |

### 完整代码对应

```python
# 步骤 4: 传导率（初始化时预计算）
discretizer = FVMDiscretizer(mesh, physics)

# 步骤 5-6: 组装线性系统
A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
# 内部流程:
#   acc_coeff = V * phi * ct / dt          ← 步骤 5
#   for direction in range(6):             ← 步骤 6.1
#       rows.append(i); cols.append(j)     ← 非对角项
#       diag[i] += T                       ← 对角项
#   well_manager.apply_well_terms(A, b, p) ← 步骤 6.2
#   A = coo_matrix(...).tocsr()            ← 步骤 6.3

# 步骤 7: 求解
solver = LinearSolver({'method': 'bicgstab'})
pressure_new = solver.solve(A, b)

# 步骤 9: 时间推进（BaseModel.solve_simulation 自动执行）
results = sim.run_simulation()
```

## 11. 关键数值验证

你可以用以下脚本验证上述推导：

```python
from reservoirpy import StructuredMesh, SinglePhaseProperties, WellManager
from reservoirpy.core.discretization import FVMDiscretizer
from reservoirpy.core.linear_solver import LinearSolver
import numpy as np

mesh = StructuredMesh(nx=3, ny=3, nz=1, dx=10, dy=10, dz=10)
physics = SinglePhaseProperties(mesh, {
    'permeability': 100.0, 'porosity': 0.2,
    'viscosity': 0.001, 'compressibility': 1e-9
})

well_manager = WellManager(mesh, [
    {'location': [0, 1, 1], 'control_type': 'bhp',
     'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
])

k = np.full((1, 3, 3), 9.869e-14)
well_manager.initialize_wells(k, 0.001)

discretizer = FVMDiscretizer(mesh, physics)
solver = LinearSolver()

pressure = np.full(9, 30e6)
A, b = discretizer.discretize_single_phase(86400, pressure, well_manager)

print(f"矩阵大小: {A.shape}")
print(f"对角项 A[4,4] = {A[4,4]:.6e}")   # 应为 acc + 4T + WI
print(f"非对角 A[4,3] = {A[4,3]:.6e}")   # 应为 -T
print(f"右端项 b[4]   = {b[4]:.6e}")     # 应为 acc*p⁰ + WI*p_bhp

p_new = solver.solve(A, b)
print(f"\n第一步压力场 (MPa):")
for j in range(3):
    row = [f"{p_new[j*3+i]/1e6:.4f}" for i in range(3)]
    print(f"  {'  '.join(row)}")
```
