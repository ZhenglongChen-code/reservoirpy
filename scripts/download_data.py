"""
从云端 API 生成数据并下载到本地

用法:
    # 生成1个单相流样本并保存到本地
    python scripts/download_data.py --mode single --n-samples 1 --output-dir data/single_phase

    # 生成10个不同参数的样本
    python scripts/download_data.py --mode single --n-samples 10 --output-dir data/single_phase

    # 只生成渗透率场
    python scripts/download_data.py --mode perm --n-samples 5 --output-dir data/perm

    # 生成两相流样本
    python scripts/download_data.py --mode two-phase --n-samples 3 --output-dir data/two_phase

    # 自定义网格和模拟参数
    python scripts/download_data.py --mode single --n-samples 5 --grid-size 64 --total-time 365 --n-time-slices 10
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

API_BASE = "http://106.14.173.234:12375"


def call_api(endpoint, payload, timeout=300, api_base=API_BASE):
    req = urllib.request.Request(
        f"{api_base}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def download_and_save(signed_url, save_path):
    with urllib.request.urlopen(signed_url) as resp:
        npz_bytes = resp.read()
    with open(save_path, "wb") as f:
        f.write(npz_bytes)
    return len(npz_bytes)


def save_metadata(save_path, api_result, payload):
    meta_path = save_path.replace(".npz", "_meta.json")
    meta = {
        "api_response": api_result,
        "request_payload": payload,
        "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def generate_perm_params(seed):
    return {
        "nx": 64, "ny": 64,
        "major_range": 80 + (seed % 5) * 20,
        "minor_range": 40 + (seed % 4) * 15,
        "azimuth": (seed * 37) % 180,
        "sill": 0.5 + (seed % 5) * 0.2,
        "mean_log_perm": 1.5 + (seed % 6) * 0.2,
        "std_log_perm": 0.3 + (seed % 5) * 0.1,
        "seed": seed,
    }


def generate_single_phase_params(seed, grid_size=64, total_time=365, n_slices=10):
    rng = np.random.RandomState(seed)
    n_wells = rng.randint(2, 5)
    wells = []
    for _ in range(n_wells):
        wells.append({
            "x": int(rng.randint(0, grid_size)),
            "y": int(rng.randint(0, grid_size)),
            "is_injector": len(wells) == 0,
            "bhp_MPa": round(rng.uniform(20, 40), 1),
        })
    return {
        "nx": grid_size, "ny": grid_size,
        "perm_seed": seed,
        "major_range": 80 + (seed % 5) * 20,
        "minor_range": 40 + (seed % 4) * 15,
        "azimuth": (seed * 37) % 180,
        "mean_log_perm": 1.5 + (seed % 6) * 0.2,
        "std_log_perm": 0.3 + (seed % 5) * 0.1,
        "viscosity_mPas": round(rng.uniform(0.5, 5.0), 2),
        "porosity": round(rng.uniform(0.1, 0.3), 2),
        "initial_pressure_MPa": round(rng.uniform(20, 40), 1),
        "total_time_days": total_time,
        "n_time_slices": n_slices,
        "wells": wells,
    }


def generate_two_phase_params(seed, grid_size=64, total_time=365, n_slices=10):
    params = generate_single_phase_params(seed, grid_size, total_time, n_slices)
    params["oil_viscosity_mPas"] = round(np.random.uniform(2, 10), 2)
    params["water_viscosity_mPas"] = round(np.random.uniform(0.3, 2), 2)
    params["initial_saturation"] = round(np.random.uniform(0.1, 0.3), 2)
    for key in ["viscosity_mPas", "porosity"]:
        params.pop(key, None)
    return params


def main():
    parser = argparse.ArgumentParser(description="从云端 API 生成数据并下载到本地")
    parser.add_argument("--mode", choices=["perm", "single", "two-phase"],
                        default="single", help="数据类型")
    parser.add_argument("--n-samples", type=int, default=1, help="生成样本数")
    parser.add_argument("--start-seed", type=int, default=0, help="起始随机种子")
    parser.add_argument("--output-dir", type=str, default=None, help="保存目录")
    parser.add_argument("--grid-size", type=int, default=64, help="网格大小")
    parser.add_argument("--total-time", type=float, default=365, help="模拟天数")
    parser.add_argument("--n-time-slices", type=int, default=10, help="时间切片数")
    parser.add_argument("--api-base", type=str, default=API_BASE, help="API地址")
    args = parser.parse_args()

    api_base = args.api_base or API_BASE

    if args.output_dir is None:
        args.output_dir = f"data/{args.mode}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_map = {
        "perm": ("/generate-permeability", generate_perm_params),
        "single": ("/simulate/single-phase", generate_single_phase_params),
        "two-phase": ("/simulate/two-phase", generate_two_phase_params),
    }
    endpoint, param_fn = mode_map[args.mode]

    print(f"模式: {args.mode}")
    print(f"样本数: {args.n_samples}")
    print(f"保存目录: {output_dir.resolve()}")
    print(f"API: {api_base}")
    print("=" * 60)

    success = 0
    failed = 0
    total_bytes = 0

    for i in range(args.n_samples):
        seed = args.start_seed + i
        print(f"\n[{i+1}/{args.n_samples}] seed={seed}")

        try:
            if args.mode == "perm":
                payload = param_fn(seed)
            elif args.mode == "single":
                payload = param_fn(seed, args.grid_size, args.total_time, args.n_time_slices)
            else:
                payload = param_fn(seed, args.grid_size, args.total_time, args.n_time_slices)

            t0 = time.time()
            result = call_api(endpoint, payload, api_base=api_base)
            api_time = time.time() - t0

            if "download_url" not in result:
                print(f"  ❌ 无下载URL (storage={result.get('storage')})")
                failed += 1
                continue

            save_name = f"{args.mode}_seed{seed:04d}.npz"
            save_path = str(output_dir / save_name)

            dl_size = download_and_save(result["download_url"], save_path)
            save_metadata(save_path, result, payload)
            total_bytes += dl_size
            success += 1

            print(f"  ✅ {save_name} ({dl_size/1024:.1f}KB) "
                  f"API耗时={result.get('elapsed_seconds', 0):.1f}s "
                  f"总耗时={api_time:.1f}s")

            if "pressure_stats" in result:
                ps = result["pressure_stats"]
                print(f"     压力: {ps['min_MPa']:.2f} ~ {ps['max_MPa']:.2f} MPa")
            if "perm_stats" in result:
                ps = result["perm_stats"]
                print(f"     渗透率: log_mean={ps['log_mean']:.2f}, log_std={ps['log_std']:.2f}")

        except urllib.error.URLError as e:
            print(f"  ❌ 网络错误: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"完成! 成功={success}, 失败={failed}, 总大小={total_bytes/1024/1024:.1f}MB")
    print(f"数据保存在: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
