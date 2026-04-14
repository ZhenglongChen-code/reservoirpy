# 井模型（Peaceman 模型）

## 基本公式

Peaceman 井模型将井的产量与井底流压和网格压力联系起来：

$$Q = WI \cdot (p_{\text{cell}} - p_{\text{bhp}})$$

其中 $WI$ 是井指数（Well Index），$p_{\text{cell}}$ 是井所在网格单元的压力，$p_{\text{bhp}}$ 是井底流压。

## 井指数计算

### 等效半径

$$r_e = 0.14 \sqrt{\Delta x^2 + \Delta y^2}$$

对于各向异性渗透率：

$$r_e = 0.28 \frac{\sqrt[4]{(k_x/k_y) \Delta y^2 + (k_y/k_x) \Delta x^2}}{\sqrt{k_x/k_y + k_y/k_x}}$$

### 井指数

$$WI = \frac{2\pi \sqrt{k_x k_y} \Delta z}{\mu \left[\ln(r_e/r_w) + S\right]}$$

| 参数 | 含义 | 单位 |
|------|------|------|
| $k_x, k_y$ | x/y 方向渗透率 | m² |
| $\Delta z$ | 射孔段厚度 | m |
| $\mu$ | 流体粘度 | Pa·s |
| $r_e$ | 等效半径 | m |
| $r_w$ | 井筒半径 | m |
| $S$ | 表皮因子 | — |

## 控制方式

### 定井底流压（BHP 控制）

给定 $p_{\text{bhp}}$，计算流量：

$$Q = WI \cdot (p_{\text{cell}} - p_{\text{bhp}})$$

- 当 $p_{\text{cell}} > p_{\text{bhp}}$ 时，$Q > 0$（生产井）
- 当 $p_{\text{cell}} < p_{\text{bhp}}$ 时，$Q < 0$（注入井）

### 定流量控制

给定 $Q$，反算井底流压：

$$p_{\text{bhp}} = p_{\text{cell}} - \frac{Q}{WI}$$

## 在 FVM 系统中的处理

井项作为源汇项加入线性系统：

- **对角项**：$A_{ii} += WI$
- **右端项**：$b_i += WI \cdot p_{\text{bhp}}$

```python
from reservoirpy.core.well_model import Well, WellManager

well = Well(location=[0, 5, 5], control_type='bhp', value=1e6)
well.compute_well_index(mesh, permeability, viscosity)

q = well.compute_well_term(pressure)  # Q = WI * (p_cell - p_bhp)
```
