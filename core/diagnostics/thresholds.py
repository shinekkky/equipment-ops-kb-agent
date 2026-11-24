"""报警阈值表：逐条摘自 data/manuals 下的三份手册。

为什么阈值不许写在检测代码里
--------------------------
现场运维里，"阈值"和"处置步骤"必须出自同一份文件。手册第 2.1 节写"85 ℃ 报警"，
检测代码就得按 85 ℃ 判——否则报告里会出现"已越限"和"该怎么处理"对不上的情况，
这种报告在现场是没人敢照着执行的。所以所有数字集中在本模块，每条都标注出处章节。

分级沿用手册口径：
    预警 —— 加强监视并安排检查
    报警 —— 立即处置
    停机 —— 必须停机，严禁带病运行
"""

from __future__ import annotations

from dataclasses import dataclass

NORMAL = "正常"
ATTENTION = "注意"  # 未越限的早期异常，由 Z-score / 趋势 / 多变量三层给出
WARN = "预警"
ALARM = "报警"
TRIP = "停机"

# 严重度只有大小关系有意义，用它给多层的发现排序、合并。
# 「注意」排在预警之前：它表示"还没到手册的预警线，但已经在变坏"，
# 这类结论必须和越限区分开，否则报告会把隐患说成事故、把事故说成隐患。
SEVERITY_ORDER = {NORMAL: 0, ATTENTION: 1, WARN: 2, ALARM: 3, TRIP: 4}


@dataclass(frozen=True)
class Band:
    """一个监测指标的分级阈值。

    direction="high" 表示越大越危险（温度、振动、电流）；
    direction="low"  表示越小越危险（出口压力偏低、光功率过低）。

    warn / alarm / trip 允许为 None —— 手册没给某一级时不许自己编一个，
    ISO 10816 的振动分区就只有三条界线，硬凑出的"预警线"会让报告凭空多出告警。
    """

    metric: str
    label: str
    unit: str
    direction: str
    warn: float | None
    alarm: float | None
    trip: float | None
    source: str

    def evaluate(self, value: float | None) -> str:
        """判定单个采样点的严重度。空值与 NaN 一律视为正常。"""
        if value is None or value != value:  # NaN != NaN
            return NORMAL

        # 从严到宽依次比较，命中即返回，保证停机优先于报警优先于预警
        for name, edge in ((TRIP, self.trip), (ALARM, self.alarm), (WARN, self.warn)):
            if edge is None:
                continue
            if self.direction == "high" and value > edge:
                return name
            if self.direction == "low" and value < edge:
                return name
        return NORMAL

    def boundary_for(self, severity: str) -> float | None:
        """取某一级的界线值，用于报告里写"越限（界线 85 ℃）"。"""
        return {TRIP: self.trip, ALARM: self.alarm, WARN: self.warn}.get(severity)


# ---------------------------------------------------------------- 离心泵
# 出处：《离心泵故障排查手册》2.1 运行参数报警阈值
# 注意电流阈值与设备容量强相关，必须按型号套用：把 CXB-200 的 52 A 用在
# 额定 88 A 的 CXB-350 上，会导致该泵全程误报——手册专门用一段提醒过这件事。

_PUMP_TEMP = Band("温度_℃", "轴承温度", "℃", "high", 75.0, 85.0, 95.0, "离心泵故障排查手册 2.1")
_PUMP_PRESS = Band("压力_MPa", "出口压力", "MPa", "low", 0.38, 0.32, 0.25, "离心泵故障排查手册 2.1")
_PUMP_VIB = Band("振动_mm_s", "振动烈度", "mm/s", "high", 4.5, 7.1, 11.0, "离心泵故障排查手册 2.1")

_PUMP_CURRENT = {
    "CXB-200": Band("电流_A", "电机电流", "A", "high", 48.0, 52.0, 58.0, "离心泵故障排查手册 2.1"),
    "CXB-350": Band("电流_A", "电机电流", "A", "high", 94.0, 102.0, 113.0, "离心泵故障排查手册 2.1"),
}

# ---------------------------------------------------------------- 旋转设备
# 出处：《旋转设备振动诊断与润滑管理规范》4.3 温升限值 / 2.2 转速与振动限值
# M-2xx 属第 2 组（中型机组，15~75 kW），振动分区界线 A/B=2.8、B/C=7.1、C/D=11.0。
# 这里只取后两条：进入 B 区是旧机的正常状态，把它当预警会让每台设备常年挂着告警；
# 而 B/C 界对应"安排计划检修"、C/D 界对应"立即停机"，才是真正要出手的点。
# 振动早期劣化由 detectors 的趋势层负责提前发现，不靠压低下限来实现。

_ROT_TEMP = Band("温度_℃", "轴承温度", "℃", "high", 70.0, 85.0, None, "旋转设备振动诊断与润滑管理规范 4.3")
_ROT_VIB = Band("振动_mm_s", "振动烈度", "mm/s", "high", None, 7.1, 11.0, "旋转设备振动诊断与润滑管理规范 2.2")

# ---------------------------------------------------------------- 网络设备
# 出处：《网络设备告警处置指南》2.2 性能指标阈值

_NET_CPU = Band("CPU利用率_%", "CPU 利用率", "%", "high", 60.0, 80.0, None, "网络设备告警处置指南 2.2")
_NET_MEM = Band("内存利用率_%", "内存利用率", "%", "high", 70.0, 85.0, None, "网络设备告警处置指南 2.2")
_NET_LOSS = Band("端口丢包率_%", "端口丢包率", "%", "high", 0.1, 1.0, None, "网络设备告警处置指南 2.2")
_NET_TEMP = Band("温度_℃", "光模块温度", "℃", "high", 60.0, 70.0, None, "网络设备告警处置指南 2.2")


BANDS: dict[str, list[Band]] = {
    "离心泵": [_PUMP_TEMP, _PUMP_PRESS, _PUMP_VIB],
    "旋转设备": [_ROT_TEMP, _ROT_VIB],
    "网络交换机": [_NET_TEMP, _NET_CPU, _NET_MEM, _NET_LOSS],
}

# 多变量检测用的特征组合。
# 旋转设备取"振动 + 温度 + 电流"三参数，出自规范第 5 章状态检修建议：
# "建立振动 + 温度 + 电流三参数联合基线，单参数正常但三参数同步偏移时……"
MULTIVARIATE_FEATURES: dict[str, list[str]] = {
    "离心泵": ["温度_℃", "压力_MPa", "振动_mm_s", "电流_A"],
    "旋转设备": ["振动_mm_s", "温度_℃", "电流_A"],
    "网络交换机": ["CPU利用率_%", "内存利用率_%", "端口丢包率_%"],
}

# 各设备类型对应的手册，用于生成检索提示
MANUAL_FOR_TYPE: dict[str, str] = {
    "离心泵": "离心泵故障排查手册",
    "旋转设备": "旋转设备振动诊断与润滑管理规范",
    "网络交换机": "网络设备告警处置指南",
}


def bands_for(device_type: str, model: str) -> list[Band]:
    """取某台设备适用的阈值列表。未知类型返回空列表，由上层跳过。"""
    bands = list(BANDS.get(device_type, []))
    if device_type == "离心泵":
        current = _PUMP_CURRENT.get(model)
        if current is not None:
            bands.append(current)
    return bands


def band_for(device_type: str, model: str, metric: str) -> Band | None:
    for band in bands_for(device_type, model):
        if band.metric == metric:
            return band
    return None


def metric_for_label(label: str) -> str | None:
    """把中文指标名换回列名。

    运维人员说的是"轴承温度""振动烈度"，代码里要的是"温度_℃""振动_mm_s"。
    工具层接收自然语言参数，必须在这一步把两者接上。
    """
    # 按去空格后的形式比对：手册和界面里写的是"CPU 利用率"（带空格），
    # 而人打字时几乎不会加那个空格，严格匹配会让工具白白失败。
    needle = label.replace(" ", "").strip()
    for bands in BANDS.values():
        for band in bands:
            if band.label.replace(" ", "") == needle:
                return band.metric
    if needle == "电机电流":
        return "电流_A"
    return None


def feature_labels(features: list[str]) -> list[str]:
    """把列名换成中文标签，供报告与归因说明使用。"""
    lookup = {
        "温度_℃": "轴承温度",
        "压力_MPa": "出口压力",
        "振动_mm_s": "振动烈度",
        "电流_A": "电机电流",
        "CPU利用率_%": "CPU 利用率",
        "内存利用率_%": "内存利用率",
        "端口丢包率_%": "端口丢包率",
    }
    return [lookup.get(f, f) for f in features]
