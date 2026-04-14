"""验证 OSS 数据下载和内容检查的示例脚本"""
import numpy as np
import json
import urllib.request
import io

API_BASE = "http://106.14.173.234:12375"


def call_api(endpoint, payload):
    req = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_npz(signed_url):
    with urllib.request.urlopen(signed_url) as resp:
        return np.load(io.BytesIO(resp.read()))


def verify_perm_data():
    print("=" * 60)
    print("1. 渗透率场生成 + 下载验证")
    print("=" * 60)

    result = call_api("/generate-permeability", {
        "nx": 32, "ny": 32, "seed": 42,
        "major_range": 120, "minor_range": 80,
        "mean_log_perm": 2.0, "std_log_perm": 0.6,
    })

    print(f"  网格: {result['grid']}")
    print(f"  渗透率: {result['perm_min_mD']:.1f} ~ {result['perm_max_mD']:.1f} mD")
    print(f"  存储方式: {result['storage']}")
    print(f"  耗时: {result['elapsed_seconds']:.2f}s")

    data = download_npz(result["download_url"])
    print(f"\n  NPZ 数组: {list(data.keys())}")
    for key in data:
        arr = data[key]
        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.2f}, {arr.max():.2f}]")

    perm = data["permeability_mD"]
    log_perm = np.log10(perm)
    assert perm.shape == (32, 32), f"shape 不对: {perm.shape}"
    assert abs(log_perm.mean() - 2.0) < 1.0, f"log mean 偏差太大: {log_perm.mean()}"
    print(f"\n  ✅ 渗透率场验证通过! shape={perm.shape}, "
          f"log_mean={log_perm.mean():.3f}, log_std={log_perm.std():.3f}")


def verify_single_phase_data():
    print("\n" + "=" * 60)
    print("2. 单相流模拟 + 下载验证")
    print("=" * 60)

    result = call_api("/simulate/single-phase", {
        "nx": 32, "ny": 32, "perm_seed": 42,
        "total_time_days": 180, "n_time_slices": 5,
        "wells": [
            {"x": 16, "y": 16, "is_injector": True, "bhp_MPa": 35},
            {"x": 0, "y": 0, "is_injector": False, "bhp_MPa": 25},
        ],
    })

    print(f"  网格: {result['grid']}")
    print(f"  时间切片: {result['time_fractions']}")
    print(f"  压力: {result['pressure_stats']['min_MPa']:.2f} ~ "
          f"{result['pressure_stats']['max_MPa']:.2f} MPa")
    print(f"  存储方式: {result['storage']}")

    data = download_npz(result["download_url"])
    print(f"\n  NPZ 数组: {list(data.keys())}")
    for key in data:
        arr = data[key]
        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")

    pressure = data["pressure_MPa"]
    perm = data["permeability_mD"]
    assert pressure.shape[0] == 5, f"时间切片数不对: {pressure.shape[0]}"
    assert pressure.shape[1:] == (32, 32), f"空间维度不对: {pressure.shape[1:]}"
    assert perm.shape == (32, 32), f"渗透率 shape 不对: {perm.shape}"
    print(f"\n  ✅ 单相流数据验证通过! "
          f"pressure shape={pressure.shape}, perm shape={perm.shape}")

    print(f"\n  各时间步压力范围:")
    for i, tf in enumerate(result["time_fractions"]):
        p = pressure[i]
        print(f"    t_frac={tf:.3f}: P=[{p.min():.2f}, {p.max():.2f}] MPa, "
              f"mean={p.mean():.2f} MPa")


if __name__ == "__main__":
    verify_perm_data()
    verify_single_phase_data()
    print("\n" + "=" * 60)
    print("全部验证通过! ✅")
    print("=" * 60)
