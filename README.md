# jumpjump-autoclicker

微信“跳一跳”桌面自动点击辅助脚本，基于 OpenCV 识别棋子和目标位置，再用 pyautogui 控制鼠标按压。

## 目录结构

```text
.
├── assets/templates/          # 识别模板资源
│   ├── background/            # 游戏窗口背景模板
│   └── player/                # 棋子模板
├── src/jumpjump_autoclicker/  # 主程序包
│   ├── app.py                 # 主流程编排
│   ├── assets.py              # 模板加载
│   ├── config.py              # 配置和路径
│   ├── controller.py          # 截屏和鼠标控制
│   ├── debug.py               # 调试图片输出
│   └── vision.py              # OpenCV 识别逻辑
├── run.py                     # 推荐启动入口
└── 1.py                       # 兼容旧入口
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```powershell
python run.py
```

运行后把微信跳一跳窗口放到前台，程序会根据 `assets/templates/background` 自动识别游戏区域。

## 调参

主要参数在 `src/jumpjump_autoclicker/config.py`：

- `press_coefficient`：距离到按压时间的换算系数，效果差时优先调它。
- `background_match_threshold`：背景模板匹配阈值。
- `player_match_threshold`：棋子模板匹配阈值。
- `target_min_area` / `target_cluster_count`：目标块轮廓识别参数。
- `target_top_y_ratio` / `target_top_band_ratio`：从轮廓顶部估算落脚点的位置。
- `vertical_distance_weight`：距离计算里的纵向权重。
- `auto_adjust_coefficient`：跳跃失败后是否自动微调按压系数。
- `side_search_margin` / `current_platform_exclusion_radius`：限制目标搜索到棋子相反侧，并排除当前平台附近。

调试图片会输出到 `debug/`，该目录已加入 `.gitignore`。`target_shape_*.png` 会画出前几个候选目标和分数，蓝色粗框是最终选择。
