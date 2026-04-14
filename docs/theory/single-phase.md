# 单相流方程

## 控制方程

微可压缩单相流的控制方程由连续性方程和达西定律联立得到：

$$\phi c_t \frac{\partial p}{\partial t} = \nabla \cdot \left(\frac{\mathbf{k}}{\mu} \nabla p\right) + q$$

其中：

| 参数 | 含义 | 单位 |
|------|------|------|
| $\phi$ | 孔隙度 | — |
| $c_t$ | 综合压缩系数 | Pa⁻¹ |
| $p$ | 压力 | Pa |
| $\mathbf{k}$ | 渗透率张量 | m² |
| $\mu$ | 流体粘度 | Pa·s |
| $q$ | 源汇项 | s⁻¹ |

## FVM 离散化

对控制体积 $V_i$ 积分并应用散度定理：

$$V_i \phi_i c_t \frac{p_i^{n+1} - p_i^n}{\Delta t} = \sum_{j \in \mathcal{N}(i)} T_{ij}(p_j^{n+1} - p_i^{n+1}) + Q_i$$

### 累积项

$$\text{acc}_i = \frac{V_i \phi_i c_t}{\Delta t}$$

### 传导率

对于各向异性渗透率 $\mathbf{k} = \text{diag}(k_x, k_y, k_z)$，x 方向传导率为：

$$T_{x,ij} = \frac{2 k_{x,i} k_{x,j}}{k_{x,i} \Delta x_j + k_{x,j} \Delta x_i} \cdot \frac{\Delta y \Delta z}{\mu}$$

## 边界条件

### 无通量边界（默认）

边界面上传导率为零，自然满足：

$$\left. \frac{\partial p}{\partial n} \right|_{\text{boundary}} = 0$$

### 定压边界（通过井模型实现）

在边界单元设置一口定压井：

$$Q_{\text{well}} = WI \cdot (p_{\text{cell}} - p_{\text{bhp}})$$

## 稳态求解

当 $\Delta t \to \infty$ 时，累积项消失，方程退化为：

$$\sum_{j \in \mathcal{N}(i)} T_{ij}(p_j - p_i) + Q_i = 0$$

这可以通过 `SinglePhaseModel.solve_steady_state()` 直接求解。
