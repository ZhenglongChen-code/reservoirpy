"""
对比 lil_matrix 逐元素赋值 vs COO 批量构建

以 5x5 网格的单相流离散化为例，展示两种方式的差异
"""

import numpy as np
import time
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix

np.random.seed(42)

# ============================================================
# 场景：构建一个 7 对角的稀疏矩阵（模拟油藏模拟的 FVM 离散化）
# 对角线 + 6 个方向的邻居（x+, x-, y+, y-, z+, z-）
# ============================================================

def build_with_lil(n):
    """旧方式：lil_matrix 逐元素赋值"""
    A = lil_matrix((n, n))
    b = np.zeros(n)

    for i in range(n):
        # 对角项
        A[i, i] += 10.0

        # 6 个方向的邻居（简化示例，假设都存在）
        for direction in range(6):
            neighbor = i - (direction + 1)  # 简化：用偏移模拟邻居
            if 0 <= neighbor < n:
                A[i, neighbor] -= 1.0
                A[i, i] += 1.0  # 对角补偿

        b[i] = 10.0 * 5.0  # 右端项

    A_csr = A.tocsr()
    return A_csr, b


def build_with_coo(n):
    """新方式：COO 批量构建"""
    rows = []
    cols = []
    data = []
    b = np.zeros(n)

    diag = np.full(n, 10.0)  # 先收集对角线贡献

    for i in range(n):
        for direction in range(6):
            neighbor = i - (direction + 1)
            if 0 <= neighbor < n:
                # 非对角项：直接追加到列表
                rows.append(i)
                cols.append(neighbor)
                data.append(-1.0)
                diag[i] += 1.0  # 对角补偿

        b[i] = 10.0 * 5.0

    # 对角项：一次性追加
    rows.extend(range(n))
    cols.extend(range(n))
    data.extend(diag.tolist())

    # 一次性构建 COO 矩阵，再转 CSR
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


# ============================================================
# 1. 正确性验证：两种方式结果一致
# ============================================================
n = 25
A_lil, b_lil = build_with_lil(n)
A_coo, b_coo = build_with_coo(n)

print("=" * 60)
print("1. 正确性验证")
print("=" * 60)
print(f"lil 和 coo 结果一致: {np.allclose(A_lil.toarray(), A_coo.toarray())}")
print(f"b 一致: {np.allclose(b_lil, b_coo)}")
print()

# ============================================================
# 2. 内存对比
# ============================================================
print("=" * 60)
print("2. 内存占用对比")
print("=" * 60)

import sys
n = 1000

# lil_matrix 中间态
A_lil_mid = lil_matrix((n, n))
for i in range(n):
    A_lil_mid[i, i] = 10.0
    if i > 0:
        A_lil_mid[i, i-1] = -1.0
    if i < n - 1:
        A_lil_mid[i, i+1] = -1.0

lil_bytes = sys.getsizeof(A_lil_mid) + sys.getsizeof(A_lil_mid.rows) + sys.getsizeof(A_lil_mid.data)
print(f"lil_matrix 中间态内存: ~{lil_bytes / 1024:.1f} KB")

# COO 中间态
rows_list = list(range(n)) + list(range(1, n)) + list(range(n-1))
cols_list = list(range(n)) + list(range(n-1)) + list(range(1, n))
data_list = [10.0]*n + [-1.0]*(n-1) + [-1.0]*(n-1)

coo_bytes = (sys.getsizeof(rows_list) + sys.getsizeof(cols_list) + 
             sys.getsizeof(data_list))
print(f"COO 列表中间态内存:   ~{coo_bytes / 1024:.1f} KB")

# 最终 CSR
A_csr = A_lil_mid.tocsr()
csr_bytes = A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes
print(f"最终 CSR 内存:        ~{csr_bytes / 1024:.1f} KB")
print()

# ============================================================
# 3. 性能对比：不同规模
# ============================================================
print("=" * 60)
print("3. 性能对比（构建时间）")
print("=" * 60)

for n in [100, 1000, 5000, 10000]:
    # lil_matrix
    t0 = time.perf_counter()
    A_lil, b_lil = build_with_lil(n)
    t_lil = time.perf_counter() - t0

    # COO
    t0 = time.perf_counter()
    A_coo, b_coo = build_with_coo(n)
    t_coo = time.perf_counter() - t0

    speedup = t_lil / t_coo if t_coo > 0 else float('inf')
    print(f"n={n:>6d}:  lil={t_lil:.4f}s  coo={t_coo:.4f}s  加速比={speedup:.1f}x")

print()

# ============================================================
# 4. 为什么 lil 慢？—— 底层原理
# ============================================================
print("=" * 60)
print("4. 底层原理对比")
print("=" * 60)
print("""
lil_matrix 内部结构（每行一个链表）:
  rows[0] = [0, 1, 2]       ← 第0行的列索引
  data[0] = [10, -1, -1]    ← 第0行的数据值

  每次 A[i, j] = value 时:
  1. 找到 rows[i] 链表
  2. 遍历链表，找到列 j 应该插入的位置  ← O(k)，k是该行非零元素数
  3. 插入元素，移动后面的元素           ← O(k)
  
  总复杂度: O(n × k)，k 平均约7 → 看起来不大
  但 Python 循环 + 链表操作的开销是 C 向量化操作的 100-1000 倍！

COO 矩阵构建方式:
  rows = [0, 0, 0, 1, 1, 1, ...]   ← 所有行索引
  cols = [0, 1, 2, 0, 1, 2, ...]   ← 所有列索引
  data = [10, -1, -1, -1, 10, -1, ...] ← 所有数据值

  每次 append: O(1) amortized
  最后 coo_matrix((data, (rows, cols))): 一次性排序+构建 → O(nnz log nnz)
  
  总复杂度: O(nnz) + O(nnz log nnz)
  关键优势：append 是纯 Python 列表操作，极快；
           最终构建是 C 层面的向量化操作，也极快。
""")

# ============================================================
# 5. 实际油藏模拟场景
# ============================================================
print("=" * 60)
print("5. 实际油藏模拟场景（10x10x5 网格）")
print("=" * 60)

nx, ny, nz = 10, 10, 5
n_cells = nx * ny * nz  # 500 个单元

def build_reservoir_lil(nx, ny, nz):
    """模拟油藏 FVM 离散化 - lil 版"""
    n = nx * ny * nz
    A = lil_matrix((n, n))
    b = np.zeros(n)

    for idx in range(n):
        k = idx % nx
        j = (idx // nx) % ny
        i = idx // (nx * ny)

        A[idx, idx] += 10.0  # 累积项

        # x- 方向
        if k > 0:
            A[idx, idx - 1] -= 1.0
            A[idx, idx] += 1.0
        # x+ 方向
        if k < nx - 1:
            A[idx, idx + 1] -= 1.0
            A[idx, idx] += 1.0
        # y- 方向
        if j > 0:
            A[idx, idx - nx] -= 1.0
            A[idx, idx] += 1.0
        # y+ 方向
        if j < ny - 1:
            A[idx, idx + nx] -= 1.0
            A[idx, idx] += 1.0
        # z- 方向
        if i > 0:
            A[idx, idx - nx * ny] -= 1.0
            A[idx, idx] += 1.0
        # z+ 方向
        if i < nz - 1:
            A[idx, idx + nx * ny] -= 1.0
            A[idx, idx] += 1.0

        b[idx] = 50.0

    return A.tocsr(), b


def build_reservoir_coo(nx, ny, nz):
    """模拟油藏 FVM 离散化 - COO 版"""
    n = nx * ny * nz
    rows = []
    cols = []
    data = []
    b = np.full(n, 50.0)

    diag = np.full(n, 10.0)

    for idx in range(n):
        k = idx % nx
        j = (idx // nx) % ny
        i = idx // (nx * ny)

        if k > 0:
            rows.append(idx); cols.append(idx - 1); data.append(-1.0)
            diag[idx] += 1.0
        if k < nx - 1:
            rows.append(idx); cols.append(idx + 1); data.append(-1.0)
            diag[idx] += 1.0
        if j > 0:
            rows.append(idx); cols.append(idx - nx); data.append(-1.0)
            diag[idx] += 1.0
        if j < ny - 1:
            rows.append(idx); cols.append(idx + nx); data.append(-1.0)
            diag[idx] += 1.0
        if i > 0:
            rows.append(idx); cols.append(idx - nx * ny); data.append(-1.0)
            diag[idx] += 1.0
        if i < nz - 1:
            rows.append(idx); cols.append(idx + nx * ny); data.append(-1.0)
            diag[idx] += 1.0

    rows.extend(range(n))
    cols.extend(range(n))
    data.extend(diag.tolist())

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


t0 = time.perf_counter()
A_lil, b_lil = build_reservoir_lil(nx, ny, nz)
t_lil = time.perf_counter() - t0

t0 = time.perf_counter()
A_coo, b_coo = build_reservoir_coo(nx, ny, nz)
t_coo = time.perf_counter() - t0

print(f"网格: {nx}x{ny}x{nz} = {n_cells} 单元")
print(f"lil_matrix: {t_lil:.4f}s")
print(f"COO批量:    {t_coo:.4f}s")
print(f"加速比:     {t_lil/t_coo:.1f}x")
print(f"结果一致:   {np.allclose(A_lil.toarray(), A_coo.toarray())}")
