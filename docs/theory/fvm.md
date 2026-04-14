# 有限体积法（FVM）

有限体积法是 ReservoirPy 的核心离散化方法，广泛应用于油藏数值模拟。

## 基本思想

有限体积法的核心是将偏微分方程在控制体积上积分，利用散度定理将体积分转化为面积分：

$$\int_V \frac{\partial(\phi \rho)}{\partial t} dV + \oint_{\partial V} \rho \mathbf{v} \cdot \mathbf{n} \, dA = \int_V q \, dV$$

其中：

| 符号 | 含义 |
|------|------|
| $V$ | 控制体积 |
| $\partial V$ | 控制体积边界 |
| $\phi$ | 孔隙度 |
| $\rho$ | 流体密度 |
| $\mathbf{v}$ | 达西流速 |
| $\mathbf{n}$ | 外法向量 |
| $q$ | 源汇项 |

## 离散化步骤

### 1. 达西定律

$$\mathbf{v} = -\frac{\mathbf{k}}{\mu} \nabla p$$

### 2. 面积分离散化

对每个控制体积 $V_i$，面积分离散为各面的通量之和：

$$\oint_{\partial V_i} \rho \mathbf{v} \cdot \mathbf{n} \, dA \approx \sum_{f \in \text{faces}(i)} T_f (p_j - p_i)$$

### 3. 传导率计算

两相邻单元 $i$ 和 $j$ 之间的传导率采用调和平均：

$$T_{ij} = \frac{A_{ij}}{\frac{d_i}{k_i} + \frac{d_j}{k_j}} \cdot \frac{1}{\mu}$$

其中 $A_{ij}$ 是界面面积，$d_i, d_j$ 是单元中心到界面的距离。

### 4. 时间离散化

采用后向欧拉（隐式）格式：

$$\frac{V_i \phi_i c_t (p_i^{n+1} - p_i^n)}{\Delta t} = \sum_j T_{ij}(p_j^{n+1} - p_i^{n+1}) + Q_i$$

### 5. 组装线性系统

最终得到线性系统：

$$\mathbf{A} \mathbf{p}^{n+1} = \mathbf{b}$$

其中对角元素为：

$$A_{ii} = \frac{V_i \phi_i c_t}{\Delta t} + \sum_j T_{ij}$$

非对角元素为：

$$A_{ij} = -T_{ij}$$

右端项为：

$$b_i = \frac{V_i \phi_i c_t}{\Delta t} p_i^n + Q_i$$

## 在 ReservoirPy 中的实现

FVM 离散化由 `FVMDiscretizer` 类实现：

```python
from reservoirpy.core.discretization import FVMDiscretizer

discretizer = FVMDiscretizer(mesh, physics)
A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
```

!!! note "矩阵存储格式"
    内部使用 COO 格式批量构建稀疏矩阵，最终转换为 CSR 格式供求解器使用。
    相比 `lil_matrix` 逐元素赋值，COO 方式在大规模网格上快约 **8-10 倍**。
