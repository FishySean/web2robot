"""web2robot —— 网络视频 → 机器人动作训练数据 的流水线。

模块与流水线环节的对应（详见 README.md 与 docs/PIPELINE.md）::

    quality/     ① 取景质检     原始 RGB 三分类 + 打标（已验证）
    routing/     ② 视角与运动分类 第一/第三人称、相机是否运动 → 选技术路线
    perception/  ④ 感知适配器   WiLoR / HaWoR / MoGe → 统一 clip 格式
    retarget/    ⑤ 重定向       EgoInfinity + M7 定制
    collision/   ⑥ 碰撞检测     臂-躯 / 双手 / 手指（B0–B4）
    trajectory/  ⑥ 坏帧兜底与平滑
    robots/m7/   M7 机器人定义（MJCF 生成、hand_frame 约定）
    common/      公共工具（坐标变换、MJCF 读写、视频 IO、可视化）

绝对路径只允许出现在 ``configs/paths.yaml``，通过 ``web2robot.paths.P`` 访问。
"""
__version__ = "0.1.0"
