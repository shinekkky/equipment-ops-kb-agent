"""生成仿真设备传感器时序数据与设备台账。

数据为**仿真数据**，不是任何真实设备的运行记录。生成逻辑刻意注入了若干
典型故障模式，使四层异常检测各有专属样本，便于演示与验证：

  P-101  第 20 天起轴承温度缓慢爬升      → 趋势劣化层（尚未越报警线即被发现）
  P-102  第 15 天温度尖峰冲高至 90 ℃     → 阈值规则层 + 滚动 Z-score 层
  P-103  第 12 天温度短时冲高但未越预警线 → 仅滚动 Z-score 层（阈值层漏报）
  M-201  第 22 天起振动持续上升并越报警线 → 阈值规则层，触发故障码 P103
  M-202  第 18 天起温度/振动/电流同步各偏 +1.5σ，单指标均不越限
                                        → 仅 IsolationForest 多变量层
  SW-01  第 25 天起 CPU 与端口丢包告警   → 网络设备告警分支（N201 / N202）

用法：
    python scripts/gen_sensor_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import DEVICE_REGISTRY_CSV, SENSOR_CSV  # noqa: E402

SEED = 42
HOURS = 30 * 24  # 30 天，逐小时采集
START = pd.Timestamp("2026-03-01 00:00:00")

# 每台设备的正常工况基线：(均值, 噪声标准差)
# 网络设备无压力/振动/转速量，对应位置留空（NaN），与真实状态监测平台导出一致。
DEVICE_PROFILES: dict[str, dict] = {
    "P-101": dict(
        name="1# 循环水泵", model="CXB-200", kind="离心泵", location="循环水站 A 区",
        temp=(62.0, 2.5), pressure=(0.45, 0.018), vib=(2.8, 0.35),
        rpm=(2950, 12), current=(41.0, 1.5),
    ),
    "P-102": dict(
        name="2# 循环水泵", model="CXB-200", kind="离心泵", location="循环水站 A 区",
        temp=(63.0, 2.6), pressure=(0.44, 0.020), vib=(3.0, 0.38),
        rpm=(2950, 12), current=(42.0, 1.6),
    ),
    "P-103": dict(
        name="3# 工艺泵", model="CXB-350", kind="离心泵", location="工艺装置 B 区",
        # 该泵输送温度较低的介质，正常轴承温度基线明显低于另两台，
        # 且测量噪声更小 —— 这正是其短时异常能被 Z-score 捕捉的原因。
        temp=(58.0, 1.6), pressure=(0.62, 0.024), vib=(3.2, 0.42),
        rpm=(2950, 12), current=(80.0, 2.4),
    ),
    "SW-01": dict(
        name="1# 汇聚交换机", model="S5720-28X", kind="网络交换机", location="1# 楼宇机房",
        temp=(48.0, 3.0), cpu=(35.0, 7.0), mem=(55.0, 4.5), loss=(0.02, 0.012),
    ),
    "SW-02": dict(
        name="2# 汇聚交换机", model="S5720-28X", kind="网络交换机", location="2# 楼宇机房",
        temp=(46.0, 2.8), cpu=(30.0, 6.0), mem=(52.0, 4.0), loss=(0.018, 0.010),
    ),
    "M-201": dict(
        name="1# 引风机电机", model="Y2-280M-4", kind="旋转设备", location="锅炉房",
        temp=(58.0, 3.0), vib=(2.4, 0.32), rpm=(1480, 10), current=(28.0, 1.2),
    ),
    "M-202": dict(
        name="2# 送风机电机", model="Y2-250M-2", kind="旋转设备", location="锅炉房",
        temp=(56.0, 2.5), vib=(2.2, 0.30), rpm=(2960, 12), current=(34.0, 1.3),
    ),
}

# 设备台账补充信息
REGISTRY_EXTRA = {
    "P-101": ("2019-06-12", "沈阳水泵股份有限公司", "张伟"),
    "P-102": ("2019-06-12", "沈阳水泵股份有限公司", "张伟"),
    "P-103": ("2021-03-25", "沈阳水泵股份有限公司", "王芳"),
    "SW-01": ("2020-11-08", "华为技术有限公司", "李娜"),
    "SW-02": ("2020-11-08", "华为技术有限公司", "李娜"),
    "M-201": ("2018-09-30", "湘潭电机股份有限公司", "赵强"),
    "M-202": ("2022-05-16", "湘潭电机股份有限公司", "赵强"),
}


def _series(rng: np.random.Generator, mean: float, std: float) -> np.ndarray:
    """生成带轻微日周期波动的基线序列。"""
    hours = np.arange(HOURS)
    # 昼夜环境温度波动 ±1 个标准差幅度，周期 24 h
    daily = np.sin(2 * np.pi * hours / 24.0) * std * 0.4
    return mean + daily + rng.normal(0, std, HOURS)


def _ramp(hours_index: np.ndarray, start_day: float, slope_per_day: float) -> np.ndarray:
    """从 start_day 开始按天线性累加的斜坡（之前为 0）。"""
    days = hours_index / 24.0
    return np.clip(days - start_day, 0, None) * slope_per_day


def build_device_frame(rng: np.random.Generator, dev_id: str, prof: dict) -> pd.DataFrame:
    hours = np.arange(HOURS)
    df = pd.DataFrame({"时间": START + pd.to_timedelta(hours, unit="h")})

    temp = _series(rng, *prof["temp"])
    pressure = _series(rng, *prof["pressure"]) if "pressure" in prof else np.full(HOURS, np.nan)
    vib = _series(rng, *prof["vib"]) if "vib" in prof else np.full(HOURS, np.nan)
    rpm = _series(rng, *prof["rpm"]) if "rpm" in prof else np.full(HOURS, np.nan)
    current = _series(rng, *prof["current"]) if "current" in prof else np.full(HOURS, np.nan)
    cpu = _series(rng, *prof["cpu"]) if "cpu" in prof else np.full(HOURS, np.nan)
    mem = _series(rng, *prof["mem"]) if "mem" in prof else np.full(HOURS, np.nan)
    loss = _series(rng, *prof["loss"]) if "loss" in prof else np.full(HOURS, np.nan)
    loss = np.clip(loss, 0, None)

    # ---------------------------------------------------------- 注入故障
    if dev_id == "P-101":
        # 轴承缓慢劣化：第 20 天起温度每天上升约 1.6 ℃，30 天末约 78 ℃，
        # 全程未越 85 ℃ 报警线，但已在第 20 天越过 75 ℃ 预警线。
        temp = temp + _ramp(hours, 20, 1.6)
        vib = vib + _ramp(hours, 20, 0.05)

    elif dev_id == "P-102":
        # 突发尖峰：第 15 天 06:00—16:00 温度冲高至约 92 ℃ 后回落，
        # 峰值段维持约 11 h，越 85 ℃ 报警线约 6 h。
        spike_start = 15 * 24 + 6
        temp[spike_start:spike_start + 11] += np.array(
            [12, 18, 24, 28, 29, 29, 27, 24, 19, 13, 7], dtype=float
        )
        vib[spike_start:spike_start + 11] += np.linspace(2.4, 0.3, 11)

    elif dev_id == "P-103":
        # 短时异常：第 12 天 10:00—13:00 温度冲高 +7.5 ℃，峰值约 68 ℃，
        # 全程未越 75 ℃ 预警线，阈值规则层漏报；但相对基线已达 4σ 以上，
        # 只有滚动 Z-score 能捕捉 —— 这是统计层的独立价值所在。
        brief = 12 * 24 + 10
        temp[brief:brief + 4] += np.array([5.5, 7.5, 6.5, 4.5])

    elif dev_id == "M-201":
        # 振动持续上升：第 22 天起每天 +0.75 mm/s，30 天末约 8.4 mm/s，
        # 第 27 天前后越过 7.1 mm/s 报警线，触发故障码 P103。
        vib = vib + _ramp(hours, 22, 0.75)
        temp = temp + _ramp(hours, 24, 0.45 * 1.0)  # 摩擦生热，温度同步缓升

    elif dev_id == "M-202":
        # 多变量隐性漂移：第 16 天起温度、振动、电流**同步**各偏约 +2σ。
        # 关键在于三者各自都远未越自己的阈值（温度仍 <70 ℃、振动仍 <3.5 mm/s、
        # 电流仍 <40 A），任何单指标规则都抓不到，只有多变量模型能识别这种
        # "三参数同步偏移"的早期劣化征兆（参见《旋转设备振动诊断与润滑管理规范》5.3 条）。
        temp = temp + _ramp(hours, 16, 5.0 / 14)        # 14 天累计 +5.0 ℃
        vib = vib + _ramp(hours, 16, 0.60 / 14)         # 14 天累计 +0.60 mm/s
        current = current + _ramp(hours, 16, 2.60 / 14)  # 14 天累计 +2.60 A

    elif dev_id == "SW-01":
        # 第 25 天起遭遇异常流量：CPU 与端口丢包同步攀升并越报警线。
        # CPU 约第 28 天越过 80%，丢包率约第 27 天越过 1.0%。
        cpu = cpu + _ramp(hours, 25, 9.5)
        loss = loss + _ramp(hours, 25, 0.24)
        mem = mem + _ramp(hours, 25, 1.2)

    df["设备ID"] = dev_id
    df["设备名称"] = prof["name"]
    df["设备型号"] = prof["model"]
    df["设备类型"] = prof["kind"]
    df["温度_℃"] = np.round(temp, 2)
    df["压力_MPa"] = np.round(pressure, 3)
    df["振动_mm_s"] = np.round(vib, 2)
    df["转速_rpm"] = np.round(rpm, 0)
    df["电流_A"] = np.round(current, 2)
    df["CPU利用率_%"] = np.round(np.clip(cpu, 0, 100), 1)
    df["内存利用率_%"] = np.round(np.clip(mem, 0, 100), 1)
    df["端口丢包率_%"] = np.round(loss, 3)

    # 故障代码由控制系统在**越报警线**时置位，属于确定性派生，
    # 保证数据内部自洽：未越限就没有告警码（M-202 正是这种"无码异常"）。
    df["故障代码"] = _derive_fault_codes(df, prof["kind"], prof["model"])
    df["环境温度_℃"] = np.round(20 + np.sin(2 * np.pi * hours / 24.0) * 5, 1)
    return df


def _derive_fault_codes(df: pd.DataFrame, kind: str, model: str) -> pd.Series:
    """按报警级阈值派生故障代码；同一时刻可有多个代码，用 / 连接。"""
    codes = pd.Series([[] for _ in range(len(df))], index=df.index)

    def flag(mask: pd.Series, code: str) -> None:
        for idx in df.index[mask.fillna(False)]:
            codes.at[idx].append(code)

    if kind == "网络交换机":
        flag(df["CPU利用率_%"] > 80, "N201")
        flag(df["端口丢包率_%"] > 1.0, "N202")
        flag(df["内存利用率_%"] > 85, "N203")
        flag(df["温度_℃"] > 70, "N204")
    else:
        # 电流阈值按型号区分：CXB-350 额定 88 A，套用 CXB-200 的 52 A 会全程误报。
        current_alarm = 102.0 if model == "CXB-350" else 52.0
        flag(df["温度_℃"] > 85, "P101")
        flag(df["压力_MPa"] < 0.32, "P102")
        flag(df["振动_mm_s"] > 7.1, "P103")
        flag(df["电流_A"] > current_alarm, "P105")

    return codes.apply(lambda lst: "/".join(dict.fromkeys(lst)))


def build_registry() -> pd.DataFrame:
    rows = []
    for dev_id, prof in DEVICE_PROFILES.items():
        commissioned, vendor, owner = REGISTRY_EXTRA[dev_id]
        rows.append({
            "设备ID": dev_id,
            "设备名称": prof["name"],
            "设备型号": prof["model"],
            "设备类型": prof["kind"],
            "安装位置": prof["location"],
            "投运日期": commissioned,
            "生产厂商": vendor,
            "责任人": owner,
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)

    frames = [build_device_frame(rng, dev_id, prof) for dev_id, prof in DEVICE_PROFILES.items()]
    sensor = pd.concat(frames, ignore_index=True).sort_values(["时间", "设备ID"]).reset_index(drop=True)

    col_order = [
        "时间", "设备ID", "设备名称", "设备型号", "设备类型",
        "温度_℃", "压力_MPa", "振动_mm_s", "转速_rpm", "电流_A",
        "CPU利用率_%", "内存利用率_%", "端口丢包率_%",
        "环境温度_℃", "故障代码",
    ]
    sensor = sensor[col_order]

    sensor.to_csv(SENSOR_CSV, index=False, encoding="utf-8-sig")
    registry = build_registry()
    registry.to_csv(DEVICE_REGISTRY_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 时序数据  -> {SENSOR_CSV}")
    print(f"     {len(sensor)} 行 × {sensor.shape[1]} 列，设备 {sensor['设备ID'].nunique()} 台，"
          f"{sensor['时间'].min()} ~ {sensor['时间'].max()}")
    print(f"[OK] 设备台账  -> {DEVICE_REGISTRY_CSV}（{len(registry)} 台）")

    print("\n各设备故障码统计：")
    fault = sensor[sensor["故障代码"] != ""].groupby("设备ID")["故障代码"].agg(
        lambda s: f"{len(s)} 条 ({', '.join(sorted(set(s)))})"
    )
    if fault.empty:
        print("  （无）")
    for dev_id in sensor["设备ID"].unique():
        print(f"  {dev_id}: {fault.get(dev_id, '无告警码')}")


if __name__ == "__main__":
    main()
