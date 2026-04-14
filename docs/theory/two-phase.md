# 两相流方程

## 控制方程

油水两相流需要同时求解压力方程和饱和度方程。

### 各相的质量守恒

水相：

$$\frac{\partial(\phi \rho_w S_w)}{\partial t} + \nabla \cdot (\rho_w \mathbf{v}_w) = q_w$$

油相：

$$\frac{\partial(\phi \rho_o S_o)}{\partial t} + \nabla \cdot (\rho_o \mathbf{v}_o) = q_o$$

### 达西定律（多相流）

$$\mathbf{v}_w = -\frac{k_{rw}(S_w)}{\mu_w} \mathbf{k} \nabla p_w$$

$$\mathbf{v}_o = -\frac{k_{ro}(S_w)}{\mu_o} \mathbf{k} \nabla p_o$$

### 约束条件

$$S_w + S_o = 1$$

$$p_o - p_w = p_c(S_w) \quad \text{（毛管压力）}$$

## 压力方程

忽略毛管压力（$p_c = 0$），两相合并得到压力方程：

$$\phi c_t \frac{\partial p}{\partial t} + \nabla \cdot (\lambda_t \mathbf{k} \nabla p) = q_t$$

其中总流度：

$$\lambda_t = \frac{k_{rw}}{\mu_w} + \frac{k_{ro}}{\mu_o}$$

## 饱和度方程

水相饱和度方程（分流形式）：

$$\phi \frac{\partial S_w}{\partial t} + \nabla \cdot (f_w \mathbf{v}_t) = q_w$$

其中分流函数：

$$f_w = \frac{\lambda_w}{\lambda_w + \lambda_o} = \frac{k_{rw}/\mu_w}{k_{rw}/\mu_w + k_{ro}/\mu_o}$$

## 相对渗透率模型

### Corey 模型

$$k_{rw} = k_{rw}^{\max} \left(\frac{S_w - S_{wr}}{1 - S_{wr} - S_{or}}\right)^{n_w}$$

$$k_{ro} = k_{ro}^{\max} \left(\frac{1 - S_w - S_{or}}{1 - S_{wr} - S_{or}}\right)^{n_o}$$

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $S_{wr}$ | 束缚水饱和度 | 0.2 |
| $S_{or}$ | 残余油饱和度 | 0.2 |
| $n_w$ | 水相 Corey 指数 | 2.0 |
| $n_o$ | 油相 Corey 指数 | 2.0 |

### 毛管压力模型（Brooks-Corey）

$$p_c(S_w) = p_{c0} \left(\frac{S_w - S_{wr}}{1 - S_{wr} - S_{or}}\right)^{-1/\lambda}$$

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $p_{c0}$ | 入口毛管压力 | 1000 Pa |
| $\lambda$ | 孔隙大小分布指数 | 2.0 |
