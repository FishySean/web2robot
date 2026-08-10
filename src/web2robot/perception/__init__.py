"""第③步 感知前端 —— 从视频里把手的 3D 轨迹取出来，落成下游能吃的 clip 格式。

分两层，因为它们的变更理由不同：

- :mod:`~web2robot.perception.to_clip` —— **下游的输入契约**（EgoInfinity clip 目录）。
  跟用哪个前端无关，零前端依赖，纯 numpy + json。换前端不动它，换重定向框架才动它。
- 各前端一个模块 —— :mod:`~web2robot.perception.hawor`（相机在动）、
  :mod:`~web2robot.perception.wilor` + :mod:`~web2robot.perception.moge`（相机固定）。
  它们只负责"把自家输出换算成相机系米制关节"，深度怎么来是它们各自的事。

路线怎么选是第②步的事（见 ``web2robot/routing/``）：单目深度是这条链路最硬的瓶颈，
HaWoR 在 HO-3D 上把手腕深度误差从约 11 cm 压到 0.6 cm，所以相机运动的片段必须走它。
量出这个结论的数据和脚本在 ``evidence/depth_benchmark_ho3d/``，改任何一条深度路径之前
先跑 ``tests/test_depth_benchmark.py``。

两条前端的失败方式不一样，这决定了兜底顺序：HaWoR 靠 SLAM，条件不满足时**整段失败**；
WiLoR 逐帧检测，**从不崩溃只少检出几帧**。所以 WiLoR 是兜底的那条，即使它深度更差。
"""
