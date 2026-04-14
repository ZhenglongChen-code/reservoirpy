# 理论基础

本节介绍 ReservoirPy 中使用的数值方法和物理模型的理论基础。

## 模拟流程总览

```
物理问题 → 数学模型 → 离散化 → 线性系统 → 求解 → 后处理
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
 渗流方程   PDE/ODE   FVM      Ax=b    Direct/
           +初始条件   有限体积   稀疏矩阵  Iterative
           +边界条件
```

选择对应章节了解详情：

- **[有限体积法](fvm.md)** — FVM 离散化原理
- **[单相流方程](single-phase.md)** — 微可压缩单相流模型
- **[两相流方程](two-phase.md)** — 油水两相流模型
- **[IMPES 方法](impes.md)** — 隐式压力显式饱和度方法
- **[井模型](well-model.md)** — Peaceman 井模型
