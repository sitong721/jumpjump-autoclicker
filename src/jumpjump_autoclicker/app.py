from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from .assets import load_template, load_templates
from .config import AppConfig
from .controller import DesktopController
from .debug import DebugWriter
from .telemetry import JumpTelemetry
from .vision import VisionDetector


class JumpJumpApp:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.next_press_multiplier = 1.0
        self.pending_far_target: tuple[tuple[int, int], int] | None = None
        self.last_far_target_confirmed = False
        self.pending_low_confidence_player: tuple[tuple[int, int], int] | None = None
        self.distance_calibration = self._load_distance_calibration()
        self.debug_writer = DebugWriter(self.config.debug_mode, self.config.debug_dir)
        self.telemetry = JumpTelemetry(self.config.telemetry_path, self.config.telemetry_enabled)
        if self._calibration_bucket_count() == 0:
            bootstrapped = self._bootstrap_distance_calibration_from_history()
            if bootstrapped > 0:
                print(f"已从历史数据初始化距离力度桶: {bootstrapped} 个样本")
        historical_coefficient = self._apply_historical_coefficient()
        self.controller = DesktopController(fail_safe=self.config.pyautogui_fail_safe)

        player_template = load_template(self.config.assets_dir / "player" / "player_1.png")
        background_templates = load_templates(self.config.assets_dir / "background")
        self.vision = VisionDetector(self.config, self.debug_writer, player_template, background_templates)

        print(f"模板目录: {self.config.assets_dir}")
        print(f"数据日志: {self.config.telemetry_path}")
        print(f"距离力度表: {self.config.calibration_path}")
        print(f"初始按压系数: {self.config.press_coefficient:.3f}")
        if historical_coefficient is not None:
            print(f"已根据历史数据恢复系数: {historical_coefficient:.3f}")
        print(f"加载棋子模板: {'成功' if player_template is not None else '失败'}")
        print(f"加载背景模板: {len(background_templates)} 个")

    def _apply_historical_coefficient(self) -> float | None:
        results = self.telemetry.recent_events("jump_result", self.config.max_history_samples)
        estimates: list[float] = []
        for result in results:
            progress = result.get("progress")
            press_time_ms = result.get("press_time_ms")
            if not isinstance(progress, (int, float)) or not isinstance(press_time_ms, (int, float)):
                continue
            if progress < self.config.min_valid_progress or progress > self.config.max_valid_progress:
                continue
            planned_distance = result.get("planned_distance")
            if not isinstance(planned_distance, (int, float)) or not self._is_plausible_distance(planned_distance):
                continue
            if planned_distance <= 0:
                continue
            moved_distance = result.get("moved_distance")
            if (
                isinstance(moved_distance, (int, float))
                and moved_distance < planned_distance * self.config.min_adjust_moved_ratio
            ):
                continue
            lateral_error = result.get("lateral_error")
            max_lateral_ratio = (
                0.65 if result.get("recognition_confirmed") is True else self.config.max_adjust_lateral_error_ratio
            )
            if (
                isinstance(lateral_error, (int, float))
                and lateral_error > planned_distance * max_lateral_ratio
            ):
                continue
            distance_after = result.get("distance_after")
            if (
                isinstance(distance_after, (int, float))
                and distance_after > planned_distance * self.config.max_history_restore_distance_after_ratio
            ):
                continue

            estimated = press_time_ms / planned_distance / progress
            if self.config.min_press_coefficient <= estimated <= self.config.max_initial_press_coefficient:
                estimates.append(float(estimated))

        if not estimates:
            return None

        recommended = float(np.median(estimates))
        self.config.press_coefficient = min(
            self.config.max_initial_press_coefficient,
            max(self.config.min_initial_press_coefficient, recommended),
        )
        return self.config.press_coefficient

    def select_game_region(self) -> None:
        print("=" * 50)
        print("请切换到微信跳一跳窗口")
        print(f"程序将在{self.config.game_ready_delay_seconds:.0f}秒后开始自动识别游戏区域")
        print("=" * 50)
        time.sleep(self.config.game_ready_delay_seconds)

        screenshot = self.controller.capture_screen()
        game_window = self.vision.auto_detect_game_window(screenshot)
        if game_window is None:
            print("错误：无法识别游戏区域，请检查背景模板文件是否存在且匹配")
            print("程序退出")
            sys.exit(1)

        self.controller.set_game_window(game_window)
        print(f"自动识别游戏区域: {game_window}")

    def calculate_press_time(self, distance: float) -> float:
        multiplier = self.next_press_multiplier if self.config.next_jump_compensation_enabled else 1.0
        return distance * self._coefficient_for_distance(distance) * multiplier

    def _load_distance_calibration(self) -> dict[str, Any]:
        path = self.config.calibration_path
        if not path.exists():
            return {"version": 1, "buckets": {}}
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "buckets": {}}
        if not isinstance(data, dict):
            return {"version": 1, "buckets": {}}
        buckets = data.get("buckets")
        if not isinstance(buckets, dict):
            data["buckets"] = {}
        return data

    def _save_distance_calibration(self) -> None:
        self.config.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.calibration_path.open("w", encoding="utf-8") as file:
            json.dump(self.distance_calibration, file, ensure_ascii=False, indent=2)

    def _calibration_bucket_count(self) -> int:
        buckets = self.distance_calibration.get("buckets")
        return len(buckets) if isinstance(buckets, dict) else 0

    def _bootstrap_distance_calibration_from_history(self) -> int:
        results = self.telemetry.recent_events("jump_result", self.config.max_history_samples)
        used = 0
        for result in results:
            planned_distance = result.get("planned_distance")
            press_time_ms = result.get("press_time_ms")
            distance_after = result.get("distance_after")
            moved_distance = result.get("moved_distance")
            progress = result.get("progress")
            lateral_error = result.get("lateral_error")
            success = bool(result.get("success"))
            if not all(
                isinstance(value, (int, float))
                for value in (planned_distance, press_time_ms, distance_after, moved_distance)
            ):
                continue
            if lateral_error is not None and not isinstance(lateral_error, (int, float)):
                continue
            updated = self._update_distance_calibration_from_result(
                float(planned_distance),
                float(press_time_ms),
                float(distance_after),
                float(moved_distance),
                progress if isinstance(progress, (int, float)) else None,
                float(lateral_error) if isinstance(lateral_error, (int, float)) else None,
                success,
                silent=True,
                recognition_confirmed=result.get("recognition_confirmed") is True,
            )
            if updated:
                used += 1
        return used

    def _distance_bucket_key(self, distance: float) -> str:
        bucket_size = max(1, self.config.distance_bucket_size)
        start = int(distance // bucket_size) * bucket_size
        return f"{start}-{start + bucket_size}"

    def _bucket_for_distance(self, distance: float) -> dict[str, Any] | None:
        buckets = self.distance_calibration.get("buckets")
        if not isinstance(buckets, dict):
            return None
        bucket = buckets.get(self._distance_bucket_key(distance))
        return bucket if isinstance(bucket, dict) else None

    def _valid_bucket_points(self) -> list[tuple[float, float, str, int]]:
        buckets = self.distance_calibration.get("buckets")
        if not isinstance(buckets, dict):
            return []

        points: list[tuple[float, float, str, int]] = []
        for key, bucket in buckets.items():
            if not isinstance(bucket, dict):
                continue
            coefficient = bucket.get("coefficient")
            count = bucket.get("count", 0)
            distance_from = bucket.get("distance_from")
            distance_to = bucket.get("distance_to")
            if not isinstance(coefficient, (int, float)) or not isinstance(count, int):
                continue
            if count < self.config.min_bucket_samples:
                continue
            if not isinstance(distance_from, (int, float)) or not isinstance(distance_to, (int, float)):
                continue
            center = (float(distance_from) + float(distance_to)) / 2
            points.append((center, float(coefficient), str(key), count))
        return sorted(points, key=lambda item: item[0])

    def _interpolated_coefficient_for_distance(self, distance: float) -> tuple[float, str] | None:
        points = self._valid_bucket_points()
        if not points:
            return None

        lower: tuple[float, float, str, int] | None = None
        upper: tuple[float, float, str, int] | None = None
        for point in points:
            if point[0] <= distance:
                lower = point
            if point[0] >= distance and upper is None:
                upper = point

        max_gap = self.config.max_interpolation_gap
        if lower is not None and upper is not None and lower != upper:
            span = upper[0] - lower[0]
            if 0 < span <= max_gap:
                ratio = (distance - lower[0]) / span
                coefficient = lower[1] + (upper[1] - lower[1]) * ratio
                source = f"插值 {lower[2]}:{lower[1]:.3f} -> {upper[2]}:{upper[1]:.3f}"
                return (float(coefficient), source)
        return None

    def _coefficient_for_distance(self, distance: float) -> float:
        bucket = self._bucket_for_distance(distance)
        if bucket is not None:
            coefficient = bucket.get("coefficient")
            count = bucket.get("count", 0)
            if isinstance(coefficient, (int, float)) and isinstance(count, int) and count >= self.config.min_bucket_samples:
                return float(coefficient)

        interpolated = self._interpolated_coefficient_for_distance(distance)
        if interpolated is not None:
            return interpolated[0]
        return self.config.press_coefficient

    def _print_distance_bucket_hint(self, distance: float) -> None:
        bucket_key = self._distance_bucket_key(distance)
        bucket = self._bucket_for_distance(distance)
        coefficient = bucket.get("coefficient") if bucket is not None else None
        count = bucket.get("count", 0) if bucket is not None else 0
        if isinstance(coefficient, (int, float)) and isinstance(count, int) and count >= self.config.min_bucket_samples:
            print(f"距离力度桶: {bucket_key} -> 系数 {coefficient:.3f} ({count} 个样本)")
        else:
            interpolated = self._interpolated_coefficient_for_distance(distance)
            if interpolated is not None:
                coefficient, source = interpolated
                print(f"距离力度桶: {bucket_key} 无直接样本，{source}，使用系数 {coefficient:.3f}")
            else:
                print(f"距离力度桶: {bucket_key} 暂无可用邻近样本，使用全局系数 {self.config.press_coefficient:.3f}")

    def calibrate_coefficient(self) -> None:
        print("\n=== 校准模式 ===")
        print("请手动操作游戏跳跃几次，观察实际距离")
        print("输入实际像素距离和是否成功（y/n）")

        samples: list[float] = []
        while True:
            try:
                input_str = input("输入 距离 是否成功(如: '350 y' 或 'done'结束): ").strip()
                if input_str.lower() == "done":
                    break

                parts = input_str.split()
                if len(parts) != 2:
                    print("输入格式错误")
                    continue

                distance = float(parts[0])
                success = parts[1].lower() == "y"
                if success:
                    samples.append(distance)
                    print(f"记录成功跳跃: {distance}px")
                else:
                    print("跳跃失败，跳过记录")
            except ValueError:
                print("输入格式错误")

        if samples:
            avg_distance = np.mean(samples)
            print(f"\n平均成功距离: {avg_distance:.1f}px")
            print("根据经验，一般系数在1.3-1.4之间")
            self.config.press_coefficient = float(input("请输入新的系数值: "))

    def run(
        self,
        verify_during_run: bool = False,
        verify_limit: int = 0,
        verify_resume_delay: float = 5.0,
    ) -> None:
        print("微信跳一跳自动化助手 (学习研究版)")
        print("=" * 50)

        self.select_game_region()

        print("\n准备开始游戏...")
        print("确保游戏窗口在最前面")
        print("按 Ctrl+C 停止程序")
        time.sleep(self.config.start_delay_seconds)

        loop_count = 0
        jump_count = 0
        idle_rounds = 0
        failed_jumps = 0
        consecutive_fails = 0
        verification_samples: list[dict[str, float]] = []

        try:
            while True:
                loop_count += 1
                if self.config.max_jumps > 0 and jump_count >= self.config.max_jumps:
                    print(f"已实际执行 {jump_count} 次跳跃，达到上限，自动结束")
                    break
                if idle_rounds >= self.config.max_idle_rounds:
                    print(f"连续空转 {idle_rounds} 次，自动结束")
                    break
                if self.config.max_failed_jumps > 0 and failed_jumps >= self.config.max_failed_jumps:
                    print(f"连续失败 {failed_jumps} 次，自动结束")
                    break

                print(f"\n=== 第 {loop_count} 次检测，已实际跳 {jump_count} 次 ===")

                image = self.controller.capture_game_screen()
                if image is None:
                    print("截屏失败")
                    idle_rounds += 1
                    time.sleep(1)
                    continue

                if self.vision.is_game_over_screen(image):
                    print("检测到结算页，游戏已结束，自动停止")
                    self.telemetry.record("game_over_detected", loop_count=loop_count, jump_count=jump_count)
                    break

                player_pos = self.vision.find_player_position(image)
                if player_pos is None:
                    self.pending_low_confidence_player = None
                    print("未找到棋子，等待...")
                    idle_rounds += 1
                    consecutive_fails += 1
                    self.telemetry.record(
                        "player_not_found",
                        loop_count=loop_count,
                        jump_count=jump_count,
                        consecutive_fails=consecutive_fails,
                    )
                    if consecutive_fails >= self.config.max_player_detection_fails:
                        print("连续多次未找到棋子，尝试重新识别游戏区域...")
                        if self._redetect_game_region():
                            consecutive_fails = 0
                        time.sleep(self.config.redetect_delay_seconds)
                        continue
                    time.sleep(0.5)
                    continue

                player_match_score = self.vision.last_player_match_score
                player_detection_source = self.vision.last_player_detection_source
                if (
                    player_detection_source == "template"
                    and player_match_score is not None
                    and player_match_score <= self.config.player_stable_match_threshold
                ):
                    if self._should_wait_for_player_confirmation(player_pos):
                        print(
                            "棋子模板匹配值略低，等待位置稳定确认 "
                            f"({player_match_score:.3f} <= {self.config.player_stable_match_threshold:.3f})"
                        )
                        idle_rounds += 1
                        self.telemetry.record(
                            "player_low_confidence",
                            loop_count=loop_count,
                            jump_count=jump_count,
                            player_pos=player_pos,
                            match_score=player_match_score,
                            detection_source=player_detection_source,
                        )
                        time.sleep(0.5)
                        continue
                    print(f"棋子模板匹配值略低但位置稳定，继续使用 ({player_match_score:.3f})")
                else:
                    self.pending_low_confidence_player = None

                print(f"棋子位置: {player_pos}")
                target_pos = self.vision.find_target_position(image, player_pos)
                if target_pos is None:
                    print("未找到目标，等待...")
                    idle_rounds += 1
                    self.telemetry.record(
                        "target_not_found",
                        loop_count=loop_count,
                        jump_count=jump_count,
                        player_pos=player_pos,
                        coefficient=self.config.press_coefficient,
                    )
                    time.sleep(0.5)
                    continue

                print(f"目标位置: {target_pos}")
                distance = self.vision.weighted_distance(player_pos, target_pos)
                print(f"计算距离: {distance:.1f}px")
                if self._should_wait_for_far_target_confirmation(target_pos, distance, idle_rounds):
                    print("刚经历目标丢失且当前目标过远，等待下一帧确认")
                    idle_rounds += 1
                    self.telemetry.record(
                        "jump_skipped_unstable_far_target",
                        loop_count=loop_count,
                        jump_count=jump_count,
                        player_pos=player_pos,
                        target_pos=target_pos,
                        distance=distance,
                        idle_rounds=idle_rounds,
                        coefficient=self.config.press_coefficient,
                    )
                    time.sleep(0.5)
                    continue
                if not self._is_plausible_jump(player_pos, target_pos, distance):
                    print("目标距离/方向不合理，疑似识别到干扰物，跳过本次")
                    idle_rounds += 1
                    self.telemetry.record(
                        "jump_skipped_implausible_target",
                        loop_count=loop_count,
                        jump_count=jump_count,
                        player_pos=player_pos,
                        target_pos=target_pos,
                        distance=distance,
                        coefficient=self.config.press_coefficient,
                    )
                    time.sleep(0.5)
                    continue

                active_press_multiplier = (
                    self.next_press_multiplier if self.config.next_jump_compensation_enabled else 1.0
                )
                if active_press_multiplier != 1.0:
                    print(f"短期力度补偿: x{self.next_press_multiplier:.3f}")
                self._print_distance_bucket_hint(distance)
                press_time = self.calculate_press_time(distance)
                effective_coefficient = press_time / distance if distance > 0 else self.config.press_coefficient
                if press_time > self.config.max_press_time_ms:
                    print(
                        f"按压时间 {press_time:.1f}ms 超过上限 "
                        f"{self.config.max_press_time_ms:.1f}ms，疑似目标识别错误，跳过本次"
                    )
                    idle_rounds += 1
                    self.telemetry.record(
                        "jump_skipped_press_too_long",
                        loop_count=loop_count,
                        jump_count=jump_count,
                        player_pos=player_pos,
                        target_pos=target_pos,
                        distance=distance,
                        press_time_ms=press_time,
                        coefficient=self.config.press_coefficient,
                        effective_coefficient=effective_coefficient,
                    )
                    time.sleep(0.5)
                    continue
                print(f"按压时间: {press_time:.1f}ms")

                if verify_during_run and (verify_limit <= 0 or len(verification_samples) < verify_limit):
                    sample = self._manual_point_check_for_image(
                        image,
                        player_pos,
                        target_pos,
                        f"跳前校验 #{len(verification_samples) + 1}",
                    )
                    if sample is not None:
                        sample["loop_count"] = float(loop_count)
                        sample["jump_count"] = float(jump_count)
                        sample["planned_auto_distance"] = float(distance)
                        sample["planned_press_time_ms"] = float(press_time)
                        verification_samples.append(sample)
                        self.telemetry.record(
                            "manual_point_check",
                            **sample,
                        )
                    else:
                        print("本次跳前校验已跳过，继续自动跳跃")
                    if verify_resume_delay > 0:
                        print(f"请切回微信窗口，{verify_resume_delay:.1f} 秒后执行本次跳跃...")
                        time.sleep(verify_resume_delay)
                    print("跳前校验结束，继续执行本次跳跃")

                planned_jump_count = jump_count + 1
                idle_rounds = 0
                self.pending_far_target = None
                self.pending_low_confidence_player = None
                self.telemetry.record(
                    "jump_planned",
                    loop_count=loop_count,
                    jump_count=planned_jump_count,
                    player_pos=player_pos,
                    target_pos=target_pos,
                    distance=distance,
                    press_time_ms=press_time,
                    coefficient=self.config.press_coefficient,
                    effective_coefficient=effective_coefficient,
                    press_multiplier=active_press_multiplier,
                )
                self.controller.perform_jump(
                    press_time,
                    player_pos,
                    focus_before_press=self.config.focus_game_before_jump,
                    focus_y_offset=self.config.focus_click_y_offset,
                    focus_delay_seconds=self.config.focus_delay_seconds,
                )
                jump_count = planned_jump_count
                wait_time = 1.0 + press_time / 1000 * 0.5
                print(f"等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)

                success = self._report_jump_result(
                    jump_count,
                    player_pos,
                    target_pos,
                    distance,
                    press_time,
                    active_press_multiplier,
                )
                failed_jumps = 0 if success else failed_jumps + 1
                consecutive_fails = 0
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as exc:
            print(f"\n程序出错: {exc}")
            import traceback

            traceback.print_exc()

        print(f"\n总共执行了 {jump_count} 次跳跃")
        if verification_samples:
            self._print_manual_verification_summary(verification_samples)
        print("程序结束")

    def run_manual_point_check(self) -> None:
        print("微信跳一跳识别校验模式")
        print("=" * 50)
        self.select_game_region()

        image = self.controller.capture_game_screen()
        if image is None:
            print("截屏失败")
            return

        print("\n正在自动识别当前画面...")
        auto_player_pos = self.vision.find_player_position(image)
        auto_target_pos = self.vision.find_target_position(image, auto_player_pos)

        print(f"自动识别棋子: {auto_player_pos if auto_player_pos is not None else '未找到'}")
        print(f"自动识别目标: {auto_target_pos if auto_target_pos is not None else '未找到'}")

        manual_points = self._collect_manual_points(image, auto_player_pos, auto_target_pos)
        if manual_points is None:
            print("已取消校验")
            return

        manual_player_pos, manual_target_pos = manual_points
        print(f"手动标记棋子: {manual_player_pos}")
        print(f"手动标记目标: {manual_target_pos}")

        if auto_player_pos is not None:
            player_error = self._plain_distance(auto_player_pos, manual_player_pos)
            print(f"棋子识别偏差: {player_error:.1f}px")
        if auto_target_pos is not None:
            target_error = self._plain_distance(auto_target_pos, manual_target_pos)
            print(f"目标识别偏差: {target_error:.1f}px")

        manual_distance = self.vision.weighted_distance(manual_player_pos, manual_target_pos)
        print(f"手动距离: {manual_distance:.1f}px")
        if auto_player_pos is not None and auto_target_pos is not None:
            auto_distance = self.vision.weighted_distance(auto_player_pos, auto_target_pos)
            print(f"自动距离: {auto_distance:.1f}px")
            print(f"距离偏差: {auto_distance - manual_distance:+.1f}px")
            print(f"按手动距离估算按压: {self.calculate_press_time(manual_distance):.1f}ms")
            print(f"按自动距离估算按压: {self.calculate_press_time(auto_distance):.1f}ms")

        self._save_manual_point_check_debug(
            image,
            auto_player_pos,
            auto_target_pos,
            manual_player_pos,
            manual_target_pos,
        )
        print("校验图已保存到 debug/manual_point_check_*.png")

    def run_step_check(self) -> None:
        print("微信跳一跳单步核对模式")
        print("=" * 50)
        print("每轮先手动点棋子/目标，再选择只跳一次。")
        print("建议优先选 m 用手动点跳：如果仍偏，就是力度桶/系数问题；如果手动准而自动不准，就是检测点问题。")
        self.select_game_region()

        loop_count = 0
        jump_count = 0
        try:
            while True:
                loop_count += 1
                print(f"\n=== 单步核对 #{loop_count}，已跳 {jump_count} 次 ===")

                image = self.controller.capture_game_screen()
                if image is None:
                    print("截图失败")
                    break

                if self.vision.is_game_over_screen(image):
                    print("检测到结算页，单步核对结束")
                    self.telemetry.record("game_over_detected", loop_count=loop_count, jump_count=jump_count)
                    break

                auto_player_pos = self.vision.find_player_position(image)
                auto_target_pos = self.vision.find_target_position(image, auto_player_pos)
                print(f"自动棋子: {auto_player_pos if auto_player_pos is not None else '未找到'}")
                print(f"自动目标: {auto_target_pos if auto_target_pos is not None else '未找到'}")

                sample = self._manual_point_check_for_image(
                    image,
                    auto_player_pos,
                    auto_target_pos,
                    f"单步核对 #{loop_count}",
                )
                if sample is None:
                    choice = input("本轮未标注。Enter 重新截图，q 退出: ").strip().lower()
                    if choice == "q":
                        break
                    continue

                sample["loop_count"] = float(loop_count)
                sample["jump_count"] = float(jump_count)
                self.telemetry.record("manual_point_check", **sample)

                issue = self._manual_verification_issue(sample)
                if issue:
                    print(f"本轮更像检测问题: {issue}")
                else:
                    print("本轮自动点接近手动点，可以用自动点或手动点继续验证力度。")

                manual_player_pos = (int(sample["manual_player_x"]), int(sample["manual_player_y"]))
                manual_target_pos = (int(sample["manual_target_x"]), int(sample["manual_target_y"]))
                can_use_auto = auto_player_pos is not None and auto_target_pos is not None
                if can_use_auto:
                    jump_source = "auto"
                    jump_start = auto_player_pos
                    jump_target = auto_target_pos
                else:
                    jump_source = "manual"
                    jump_start = manual_player_pos
                    jump_target = manual_target_pos

                distance = self.vision.weighted_distance(jump_start, jump_target)
                self._print_distance_bucket_hint(distance)
                press_time = self.calculate_press_time(distance)
                effective_coefficient = press_time / distance if distance > 0 else self.config.press_coefficient
                print(
                    f"本次使用 {jump_source} 点: start={jump_start}, target={jump_target}, "
                    f"distance={distance:.1f}px, press={press_time:.1f}ms"
                )

                delay = max(0.0, self.config.step_check_auto_jump_delay_seconds)
                if delay > 0:
                    print(f"手动标点完成，{delay:.1f} 秒后自动按 {jump_source} 点执行本次跳跃...")
                    time.sleep(delay)

                planned_jump_count = jump_count + 1
                self.telemetry.record(
                    "step_check_decision",
                    loop_count=loop_count,
                    jump_count=planned_jump_count,
                    decision="jump",
                    jump_source=jump_source,
                    issue=issue,
                    selected_player_pos=jump_start,
                    selected_target_pos=jump_target,
                    selected_distance=distance,
                    selected_press_time_ms=press_time,
                    effective_coefficient=effective_coefficient,
                )
                self.telemetry.record(
                    "jump_planned",
                    loop_count=loop_count,
                    jump_count=planned_jump_count,
                    player_pos=jump_start,
                    target_pos=jump_target,
                    distance=distance,
                    press_time_ms=press_time,
                    coefficient=self.config.press_coefficient,
                    effective_coefficient=effective_coefficient,
                    press_multiplier=1.0,
                    point_source=jump_source,
                )
                self.controller.perform_jump(
                    press_time,
                    jump_start,
                    focus_before_press=self.config.focus_game_before_jump,
                    focus_y_offset=self.config.focus_click_y_offset,
                    focus_delay_seconds=self.config.focus_delay_seconds,
                )
                jump_count = planned_jump_count
                wait_time = 1.0 + press_time / 1000 * 0.5
                print(f"等待 {wait_time:.1f} 秒后复检...")
                time.sleep(wait_time)
                self._report_jump_result(
                    jump_count,
                    jump_start,
                    jump_target,
                    distance,
                    press_time,
                    1.0,
                    recognition_confirmed=issue is None,
                    recognition_issue=issue,
                )

                print("本次单步核对完成，继续下一帧。按 Ctrl+C 可退出。")
        except KeyboardInterrupt:
            print("\n单步核对已中断")

        print(f"单步核对结束，共执行 {jump_count} 次跳跃")

    def _manual_point_check_for_image(
        self,
        image: np.ndarray,
        auto_player_pos: tuple[int, int] | None,
        auto_target_pos: tuple[int, int] | None,
        label: str,
    ) -> dict[str, float] | None:
        print(f"\n{label}: 请在弹出的截图上点棋子底部中心，再点目标中心/落点")
        manual_points = self._collect_manual_points(image, auto_player_pos, auto_target_pos)
        if manual_points is None:
            return None

        manual_player_pos, manual_target_pos = manual_points
        print(f"手动标记棋子: {manual_player_pos}")
        print(f"手动标记目标: {manual_target_pos}")

        manual_distance = self.vision.weighted_distance(manual_player_pos, manual_target_pos)
        sample: dict[str, float] = {
            "manual_player_x": float(manual_player_pos[0]),
            "manual_player_y": float(manual_player_pos[1]),
            "manual_target_x": float(manual_target_pos[0]),
            "manual_target_y": float(manual_target_pos[1]),
            "manual_distance": manual_distance,
        }

        if auto_player_pos is not None:
            player_error = self._plain_distance(auto_player_pos, manual_player_pos)
            sample["player_error"] = player_error
            sample["auto_player_x"] = float(auto_player_pos[0])
            sample["auto_player_y"] = float(auto_player_pos[1])
            sample["player_dx"] = float(auto_player_pos[0] - manual_player_pos[0])
            sample["player_dy"] = float(auto_player_pos[1] - manual_player_pos[1])
            print(f"棋子识别偏差: {player_error:.1f}px")
            print(f"棋子坐标偏差: dx={sample['player_dx']:+.1f}px, dy={sample['player_dy']:+.1f}px")
        if auto_target_pos is not None:
            target_error = self._plain_distance(auto_target_pos, manual_target_pos)
            sample["target_error"] = target_error
            sample["auto_target_x"] = float(auto_target_pos[0])
            sample["auto_target_y"] = float(auto_target_pos[1])
            sample["target_dx"] = float(auto_target_pos[0] - manual_target_pos[0])
            sample["target_dy"] = float(auto_target_pos[1] - manual_target_pos[1])
            print(f"目标识别偏差: {target_error:.1f}px")
            print(f"目标坐标偏差: dx={sample['target_dx']:+.1f}px, dy={sample['target_dy']:+.1f}px")

        print(f"手动距离: {manual_distance:.1f}px")
        if auto_player_pos is not None and auto_target_pos is not None:
            auto_distance = self.vision.weighted_distance(auto_player_pos, auto_target_pos)
            distance_error = auto_distance - manual_distance
            manual_press_time = self.calculate_press_time(manual_distance)
            auto_press_time = self.calculate_press_time(auto_distance)
            sample["auto_distance"] = auto_distance
            sample["distance_error"] = distance_error
            sample["manual_press_time_ms"] = manual_press_time
            sample["auto_press_time_ms"] = auto_press_time
            sample["press_time_error_ms"] = auto_press_time - manual_press_time
            print(f"自动距离: {auto_distance:.1f}px")
            print(f"距离偏差: {distance_error:+.1f}px")
            print(f"按手动距离估算按压: {manual_press_time:.1f}ms")
            print(f"按自动距离估算按压: {auto_press_time:.1f}ms")
            print(f"按压时间偏差: {auto_press_time - manual_press_time:+.1f}ms")

        issue = self._manual_verification_issue(sample)
        if issue:
            print(f"识别可疑: {issue}")
        else:
            print("识别判断: 本跳自动点和手动点接近")

        self._save_manual_point_check_debug(
            image,
            auto_player_pos,
            auto_target_pos,
            manual_player_pos,
            manual_target_pos,
        )
        print("校验图已保存到 debug/manual_point_check_*.png")
        return sample

    def _print_manual_verification_summary(self, samples: list[dict[str, float]]) -> None:
        print("\n跳前人工校验汇总")
        print("=" * 50)
        player_errors = [sample["player_error"] for sample in samples if "player_error" in sample]
        target_errors = [sample["target_error"] for sample in samples if "target_error" in sample]
        distance_errors = [sample["distance_error"] for sample in samples if "distance_error" in sample]

        if player_errors:
            print(f"棋子偏差: 平均 {np.mean(player_errors):.1f}px，最大 {np.max(player_errors):.1f}px")
        if target_errors:
            print(f"目标偏差: 平均 {np.mean(target_errors):.1f}px，最大 {np.max(target_errors):.1f}px")
        if distance_errors:
            abs_distance_errors = [abs(value) for value in distance_errors]
            print(f"距离偏差: 平均 {np.mean(distance_errors):+.1f}px，平均绝对 {np.mean(abs_distance_errors):.1f}px，最大绝对 {np.max(abs_distance_errors):.1f}px")

        suspicious = [
            (index, self._manual_verification_issue(sample))
            for index, sample in enumerate(samples, start=1)
            if self._manual_verification_issue(sample)
        ]
        if suspicious:
            print("\n可疑跳明细")
            for index, issue in suspicious:
                print(f"#{index}: {issue}")
        else:
            print("\n可疑跳明细: 未发现明显识别偏差")

        print("\n每跳完整明细")
        for index, sample in enumerate(samples, start=1):
            player_error = sample.get("player_error")
            target_error = sample.get("target_error")
            distance_error = sample.get("distance_error")
            press_time_error = sample.get("press_time_error_ms")
            loop_count = sample.get("loop_count")
            jump_count = sample.get("jump_count")
            prefix = f"#{index}"
            if loop_count is not None and jump_count is not None:
                prefix += f" 检测{loop_count:.0f}/已跳{jump_count:.0f}"
            print(prefix)
            print(
                "  棋子: "
                f"自动=({sample.get('auto_player_x', float('nan')):.0f},{sample.get('auto_player_y', float('nan')):.0f}) "
                f"手动=({sample.get('manual_player_x', float('nan')):.0f},{sample.get('manual_player_y', float('nan')):.0f}) "
                f"误差={player_error:.1f}px "
                f"dx={sample.get('player_dx', float('nan')):+.1f} dy={sample.get('player_dy', float('nan')):+.1f}"
                if player_error is not None
                else "  棋子: 自动未找到"
            )
            print(
                "  目标: "
                f"自动=({sample.get('auto_target_x', float('nan')):.0f},{sample.get('auto_target_y', float('nan')):.0f}) "
                f"手动=({sample.get('manual_target_x', float('nan')):.0f},{sample.get('manual_target_y', float('nan')):.0f}) "
                f"误差={target_error:.1f}px "
                f"dx={sample.get('target_dx', float('nan')):+.1f} dy={sample.get('target_dy', float('nan')):+.1f}"
                if target_error is not None
                else "  目标: 自动未找到"
            )
            print(
                "  距离/按压: "
                f"自动={sample.get('auto_distance', float('nan')):.1f}px "
                f"手动={sample.get('manual_distance', float('nan')):.1f}px "
                f"距离差={distance_error:+.1f}px "
                f"按压差={press_time_error:+.1f}ms"
                if distance_error is not None and press_time_error is not None
                else "  距离/按压: N/A"
            )

    @staticmethod
    def _manual_verification_issue(sample: dict[str, float]) -> str | None:
        issues: list[str] = []
        player_error = sample.get("player_error")
        target_error = sample.get("target_error")
        distance_error = sample.get("distance_error")
        if player_error is None:
            issues.append("棋子自动未找到")
        elif player_error > 12:
            issues.append(f"棋子偏差 {player_error:.1f}px")
        if target_error is None:
            issues.append("目标自动未找到")
        elif target_error > 18:
            issues.append(f"目标偏差 {target_error:.1f}px")
        if distance_error is None:
            issues.append("距离无法对比")
        elif abs(distance_error) > 18:
            direction = "自动距离偏大" if distance_error > 0 else "自动距离偏小"
            issues.append(f"{direction} {distance_error:+.1f}px")
        return "；".join(issues) if issues else None

    def _collect_manual_points(
        self,
        image: np.ndarray,
        auto_player_pos: tuple[int, int] | None,
        auto_target_pos: tuple[int, int] | None,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        window_name = "jumpjump manual check"
        points: list[tuple[int, int]] = []

        def draw() -> np.ndarray:
            view = image.copy()
            if auto_player_pos is not None:
                cv2.circle(view, auto_player_pos, 8, (0, 0, 255), 2)
                cv2.putText(
                    view,
                    "auto player",
                    (auto_player_pos[0] + 8, auto_player_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            if auto_target_pos is not None:
                cv2.circle(view, auto_target_pos, 8, (255, 0, 0), 2)
                cv2.putText(
                    view,
                    "auto target",
                    (auto_target_pos[0] + 8, auto_target_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
            for index, point in enumerate(points, start=1):
                color = (0, 255, 0) if index == 1 else (0, 255, 255)
                label = "manual player" if index == 1 else "manual target"
                cv2.circle(view, point, 7, color, -1)
                cv2.putText(
                    view,
                    label,
                    (point[0] + 8, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            cv2.putText(
                view,
                "click player bottom, then target center. ESC/Q cancel.",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 30),
                2,
            )
            return view

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
                return
            points.append((x, y))
            cv2.imshow(window_name, draw())

        print("会弹出一张当前截图：第一下点棋子底部中心，第二下点目标中心/落点；按 ESC 或 Q 取消。")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        cv2.imshow(window_name, draw())

        try:
            while len(points) < 2:
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return None
        finally:
            cv2.destroyWindow(window_name)

        return points[0], points[1]

    def _save_manual_point_check_debug(
        self,
        image: np.ndarray,
        auto_player_pos: tuple[int, int] | None,
        auto_target_pos: tuple[int, int] | None,
        manual_player_pos: tuple[int, int],
        manual_target_pos: tuple[int, int],
    ) -> None:
        debug_img = image.copy()
        if auto_player_pos is not None:
            cv2.circle(debug_img, auto_player_pos, 8, (0, 0, 255), 2)
            cv2.putText(
                debug_img,
                "auto player",
                (auto_player_pos[0] + 8, auto_player_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        if auto_target_pos is not None:
            cv2.circle(debug_img, auto_target_pos, 8, (255, 0, 0), 2)
            cv2.putText(
                debug_img,
                "auto target",
                (auto_target_pos[0] + 8, auto_target_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        cv2.circle(debug_img, manual_player_pos, 7, (0, 255, 0), -1)
        cv2.putText(
            debug_img,
            "manual player",
            (manual_player_pos[0] + 8, manual_player_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.circle(debug_img, manual_target_pos, 7, (0, 255, 255), -1)
        cv2.putText(
            debug_img,
            "manual target",
            (manual_target_pos[0] + 8, manual_target_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.line(debug_img, manual_player_pos, manual_target_pos, (0, 255, 255), 1)
        if auto_player_pos is not None and auto_target_pos is not None:
            cv2.line(debug_img, auto_player_pos, auto_target_pos, (255, 0, 0), 1)
        self.debug_writer.save("manual_point_check", debug_img)

    @staticmethod
    def _plain_distance(start: tuple[int, int], end: tuple[int, int]) -> float:
        return float(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))

    def _redetect_game_region(self) -> bool:
        screenshot = self.controller.capture_screen()
        game_window = self.vision.auto_detect_game_window(screenshot)
        if game_window is None:
            print("重新识别游戏区域失败，继续等待...")
            self.telemetry.record("game_region_redetect_failed")
            return False

        self.controller.set_game_window(game_window)
        print(f"重新识别游戏区域: {game_window}")
        self.telemetry.record("game_region_redetected", game_window=game_window)
        return True

    def _is_plausible_distance(self, distance: float) -> bool:
        return distance <= self.config.max_jump_distance

    def _should_wait_for_far_target_confirmation(
        self,
        target_pos: tuple[int, int],
        distance: float,
        idle_rounds: int,
    ) -> bool:
        self.last_far_target_confirmed = False
        if idle_rounds <= 0 or distance <= self.config.max_jump_distance_after_target_lost:
            self.pending_far_target = None
            return False

        if self.pending_far_target is None:
            self.pending_far_target = (target_pos, 1)
            return True

        previous_pos, seen_count = self.pending_far_target
        position_delta = self.vision.weighted_distance(previous_pos, target_pos)
        if position_delta <= self.config.far_target_confirmation_tolerance:
            seen_count += 1
            self.pending_far_target = (target_pos, seen_count)
        else:
            seen_count = 1
            self.pending_far_target = (target_pos, seen_count)

        if seen_count >= self.config.far_target_confirmation_rounds:
            print(f"远目标连续确认 {seen_count} 次，允许跳跃")
            self.pending_far_target = None
            self.last_far_target_confirmed = True
            return False
        return True

    def _is_plausible_jump(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        distance: float,
    ) -> bool:
        horizontal_separation = abs(target_pos[0] - start_pos[0])
        if distance < self.config.min_jump_distance:
            return (
                distance <= self.config.close_target_max_distance
                and horizontal_separation >= self.config.close_target_min_horizontal_separation
            )
        if not self._is_plausible_distance(distance):
            if not (
                self.last_far_target_confirmed
                and distance <= self.config.max_jump_distance + self.config.confirmed_far_target_distance_margin
            ):
                return False
            print(
                f"远目标已连续确认，允许超过常规距离上限 "
                f"{distance:.1f}px > {self.config.max_jump_distance:.1f}px"
            )
        if horizontal_separation < self.config.min_horizontal_separation:
            return False
        return True

    def _should_wait_for_player_confirmation(self, player_pos: tuple[int, int]) -> bool:
        required_rounds = max(1, self.config.player_low_confidence_confirmation_rounds)
        if required_rounds <= 1:
            self.pending_low_confidence_player = None
            return False

        pending = self.pending_low_confidence_player
        if pending is None:
            self.pending_low_confidence_player = (player_pos, 1)
            return True

        previous_pos, count = pending
        movement = float(
            np.sqrt((player_pos[0] - previous_pos[0]) ** 2 + (player_pos[1] - previous_pos[1]) ** 2)
        )
        if movement > self.config.player_low_confidence_position_tolerance:
            self.pending_low_confidence_player = (player_pos, 1)
            return True

        count += 1
        self.pending_low_confidence_player = (player_pos, count)
        if count < required_rounds:
            return True

        self.pending_low_confidence_player = None
        return False

    def _report_jump_result(
        self,
        jump_count: int,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        planned_distance: float,
        press_time_ms: float,
        press_multiplier: float,
        recognition_confirmed: bool = False,
        recognition_issue: str | None = None,
    ) -> bool:
        image_after = self.controller.capture_game_screen()
        if image_after is None:
            self.telemetry.record("result_capture_failed", jump_count=jump_count)
            return False

        player_pos_after = self.vision.find_player_position(image_after)
        if player_pos_after is None:
            self.telemetry.record("result_player_not_found", jump_count=jump_count)
            return False

        distance_after = np.sqrt(
            (player_pos_after[0] - target_pos[0]) ** 2 + (player_pos_after[1] - target_pos[1]) ** 2
        )
        progress = self._jump_progress(start_pos, target_pos, player_pos_after)
        lateral_error = self._jump_lateral_error(start_pos, target_pos, player_pos_after, progress)
        success = bool(distance_after <= self.config.success_distance_threshold)
        moved_distance = self.vision.weighted_distance(start_pos, player_pos_after)
        press_ms_per_px = press_time_ms / planned_distance if planned_distance > 0 else None
        estimated_required_press_ms_per_px = (
            press_time_ms / planned_distance / progress
            if planned_distance > 0 and progress is not None and progress > 0
            else None
        )
        self.telemetry.record(
            "jump_result",
            jump_count=jump_count,
            start_pos=start_pos,
            target_pos=target_pos,
            actual_pos=player_pos_after,
            planned_distance=planned_distance,
            press_time_ms=press_time_ms,
            coefficient=self.config.press_coefficient,
            press_multiplier=press_multiplier,
            press_ms_per_px=press_ms_per_px,
            estimated_required_press_ms_per_px=estimated_required_press_ms_per_px,
            distance_after=float(distance_after),
            moved_distance=moved_distance,
            lateral_error=lateral_error,
            progress=progress,
            success=success,
            recognition_confirmed=recognition_confirmed,
            recognition_issue=recognition_issue,
        )
        self._update_distance_calibration_from_result(
            planned_distance,
            press_time_ms,
            float(distance_after),
            moved_distance,
            progress,
            lateral_error,
            success,
            recognition_confirmed=recognition_confirmed,
        )

        if not success:
            print(f"跳后偏差 {distance_after:.1f}px，继续检测下一跳")
            if moved_distance < planned_distance * self.config.min_effective_moved_ratio:
                print("本次按压后棋子几乎未移动，疑似未起跳/结算遮罩/棋子复检异常，跳过力度修正")
                self._skip_next_press_multiplier("本次跳跃无有效移动")
                self.telemetry.record(
                    "jump_result_no_effective_movement",
                    jump_count=jump_count,
                    planned_distance=planned_distance,
                    press_time_ms=press_time_ms,
                    distance_after=float(distance_after),
                    moved_distance=moved_distance,
                    progress=progress,
                    coefficient=self.config.press_coefficient,
                    press_multiplier=press_multiplier,
                )
                return False
            coefficient_adjusted = self._adjust_coefficient(
                start_pos,
                target_pos,
                player_pos_after,
                planned_distance,
                float(distance_after),
                moved_distance,
                progress,
                lateral_error,
                recognition_confirmed=recognition_confirmed,
            )
            if coefficient_adjusted:
                self._skip_next_press_multiplier("全局系数已调整，避免重复加力")
            else:
                self._update_next_press_multiplier(
                    planned_distance,
                    float(distance_after),
                    moved_distance,
                    progress,
                    lateral_error,
                )
        else:
            print(f"跳跃成功：棋子距离目标 {distance_after:.1f}px")
            self._reset_next_press_multiplier()
        return success

    def _jump_progress(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        actual_pos: tuple[int, int],
    ) -> float | None:
        jump_vector = np.array([target_pos[0] - start_pos[0], target_pos[1] - start_pos[1]], dtype=np.float32)
        actual_vector = np.array([actual_pos[0] - start_pos[0], actual_pos[1] - start_pos[1]], dtype=np.float32)
        jump_length_sq = float(np.dot(jump_vector, jump_vector))
        if jump_length_sq == 0:
            return None
        return float(np.dot(actual_vector, jump_vector) / jump_length_sq)

    def _jump_lateral_error(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        actual_pos: tuple[int, int],
        progress: float | None = None,
    ) -> float | None:
        if progress is None:
            progress = self._jump_progress(start_pos, target_pos, actual_pos)
        if progress is None:
            return None

        start_vector = np.array(start_pos, dtype=np.float32)
        jump_vector = np.array([target_pos[0] - start_pos[0], target_pos[1] - start_pos[1]], dtype=np.float32)
        actual_vector = np.array(actual_pos, dtype=np.float32)
        projected = start_vector + jump_vector * progress
        return float(np.linalg.norm(actual_vector - projected))

    def _update_distance_calibration_from_result(
        self,
        planned_distance: float,
        press_time_ms: float,
        distance_after: float,
        moved_distance: float,
        progress: float | None,
        lateral_error: float | None,
        success: bool,
        silent: bool = False,
        recognition_confirmed: bool = False,
    ) -> bool:
        if planned_distance <= 0 or progress is None or progress <= 0:
            return False
        if progress < self.config.min_valid_progress or progress > self.config.max_valid_progress:
            return False
        if moved_distance < planned_distance * self.config.min_adjust_moved_ratio:
            return False
        max_lateral_ratio = 0.65 if recognition_confirmed else self.config.max_adjust_lateral_error_ratio
        if lateral_error is not None and lateral_error > planned_distance * max_lateral_ratio:
            return False
        if not success and distance_after > planned_distance * self.config.max_adjust_distance_after_ratio:
            return False

        estimated = press_time_ms / planned_distance / progress
        if estimated < self.config.min_bucket_coefficient or estimated > self.config.max_bucket_coefficient:
            return False

        bucket_key = self._distance_bucket_key(planned_distance)
        buckets = self.distance_calibration.setdefault("buckets", {})
        if not isinstance(buckets, dict):
            buckets = {}
            self.distance_calibration["buckets"] = buckets

        existing = buckets.get(bucket_key)
        if not isinstance(existing, dict):
            existing = {
                "distance_from": int(bucket_key.split("-")[0]),
                "distance_to": int(bucket_key.split("-")[1]),
                "coefficient": estimated,
                "count": 0,
            }
            buckets[bucket_key] = existing

        old_coefficient = existing.get("coefficient", estimated)
        if not isinstance(old_coefficient, (int, float)):
            old_coefficient = estimated

        count = existing.get("count", 0)
        if not isinstance(count, int):
            count = 0

        learning_rate = self.config.bucket_learning_rate
        new_coefficient = float(old_coefficient) * (1 - learning_rate) + float(estimated) * learning_rate
        existing.update(
            {
                "coefficient": new_coefficient,
                "count": count + 1,
                "last_estimated_coefficient": float(estimated),
                "last_distance": float(planned_distance),
                "last_press_time_ms": float(press_time_ms),
                "last_progress": float(progress),
                "last_distance_after": float(distance_after),
                "last_success": bool(success),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._save_distance_calibration()
        if not silent:
            print(
                f"更新距离力度桶 {bucket_key}: {float(old_coefficient):.3f} -> "
                f"{new_coefficient:.3f} (样本 {count + 1})"
            )
        return True

    def _reset_next_press_multiplier(self) -> None:
        self.next_press_multiplier = 1.0

    def _skip_next_press_multiplier(self, reason: str) -> None:
        if self.next_press_multiplier != 1.0:
            print(f"短期力度补偿重置：{reason}")
        self._reset_next_press_multiplier()

    def _update_next_press_multiplier(
        self,
        planned_distance: float,
        distance_after: float,
        moved_distance: float,
        progress: float | None,
        lateral_error: float | None,
    ) -> None:
        if not self.config.next_jump_compensation_enabled:
            return
        if progress is None or progress <= 0:
            self._skip_next_press_multiplier("跳后进度无效")
            return
        if not self._is_plausible_distance(planned_distance):
            self._skip_next_press_multiplier("目标距离不可信")
            return

        min_moved_distance = planned_distance * self.config.min_adjust_moved_ratio
        max_lateral_error = planned_distance * self.config.max_compensation_lateral_error_ratio
        max_distance_after = planned_distance * self.config.max_adjust_distance_after_ratio
        if moved_distance < min_moved_distance:
            self._skip_next_press_multiplier("棋子移动太少，疑似点击/复检异常")
            return
        if lateral_error is not None and lateral_error > max_lateral_error:
            self._skip_next_press_multiplier("横向偏差太大，疑似目标点偏了")
            return
        if 0.88 <= progress <= 1.12 and distance_after > max_distance_after:
            self._skip_next_press_multiplier("方向接近但落点仍远，疑似目标中心不可靠")
            return

        correction = 1.0 + (1.0 - progress) * self.config.next_jump_compensation_gain
        next_multiplier = min(
            self.config.max_next_press_multiplier,
            max(self.config.min_next_press_multiplier, correction),
        )
        self.next_press_multiplier = float(next_multiplier)
        direction = "跳短了，加力" if progress < 1.0 else "跳远了，减力"
        change_percent = (self.next_press_multiplier - 1.0) * 100
        print(
            f"下一跳力度补偿: x{self.next_press_multiplier:.3f} "
            f"({direction} {change_percent:+.1f}%，进度 {progress:.2f})"
        )
        self.telemetry.record(
            "next_press_multiplier_updated",
            progress=progress,
            planned_distance=planned_distance,
            distance_after=distance_after,
            moved_distance=moved_distance,
            lateral_error=lateral_error,
            next_press_multiplier=self.next_press_multiplier,
        )

    def _adjust_coefficient(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        actual_pos: tuple[int, int],
        planned_distance: float,
        distance_after: float,
        moved_distance: float,
        progress: float | None = None,
        lateral_error: float | None = None,
        recognition_confirmed: bool = False,
    ) -> bool:
        if not self.config.auto_adjust_coefficient:
            return False

        if progress is None:
            progress = self._jump_progress(start_pos, target_pos, actual_pos)
        if progress is None or progress <= 0:
            return False
        if lateral_error is None:
            lateral_error = self._jump_lateral_error(start_pos, target_pos, actual_pos, progress)
        if not self._is_plausible_distance(planned_distance):
            return False
        min_moved_distance = planned_distance * self.config.min_adjust_moved_ratio
        max_lateral_ratio = 0.65 if recognition_confirmed else self.config.max_adjust_lateral_error_ratio
        max_lateral_error = planned_distance * max_lateral_ratio
        max_distance_after = planned_distance * self.config.max_adjust_distance_after_ratio
        if moved_distance < min_moved_distance:
            self._record_coefficient_skip(
                "movement_too_small",
                progress,
                planned_distance,
                distance_after,
                moved_distance,
                lateral_error,
            )
            return False
        if lateral_error is not None and lateral_error > max_lateral_error:
            self._record_coefficient_skip(
                "lateral_error_too_large",
                progress,
                planned_distance,
                distance_after,
                moved_distance,
                lateral_error,
            )
            return False
        if 0.88 <= progress <= 1.12 and distance_after > max_distance_after:
            self._record_coefficient_skip(
                "target_center_unreliable",
                progress,
                planned_distance,
                distance_after,
                moved_distance,
                lateral_error,
            )
            return False
        if progress < self.config.min_valid_progress or progress > self.config.max_valid_progress:
            self._record_coefficient_skip(
                "progress_out_of_range",
                progress,
                planned_distance,
                distance_after,
                moved_distance,
                lateral_error,
            )
            return False

        old_coefficient = self.config.press_coefficient
        if progress < 0.92:
            target_coefficient = self.config.press_coefficient / progress
            blended = self._blend_coefficient(target_coefficient)
            minimum_step = self.config.press_coefficient * (1 + self.config.coefficient_adjust_step)
            upper_step = self.config.press_coefficient * (1 + self.config.max_coefficient_change_ratio)
            self.config.press_coefficient = min(
                self.config.max_press_coefficient,
                upper_step,
                max(blended, minimum_step),
            )
            reason = "力度偏小"
        elif progress > 1.08:
            target_coefficient = self.config.press_coefficient / progress
            blended = self._blend_coefficient(target_coefficient)
            maximum_step = self.config.press_coefficient * (1 - self.config.coefficient_adjust_step)
            lower_step = self.config.press_coefficient * (1 - self.config.max_coefficient_change_ratio)
            self.config.press_coefficient = max(
                self.config.min_press_coefficient,
                lower_step,
                min(blended, maximum_step),
            )
            reason = "力度偏大"
        else:
            return False

        if self.config.press_coefficient == old_coefficient:
            self._record_coefficient_skip(
                "coefficient_at_limit",
                progress,
                planned_distance,
                distance_after,
                moved_distance,
                lateral_error,
            )
            print(f"自动微调跳过：系数已到边界 {old_coefficient:.3f}，优先检查目标识别")
            return False

        self.telemetry.record(
            "coefficient_adjusted",
            reason=reason,
            progress=progress,
            planned_distance=planned_distance,
            distance_after=distance_after,
            moved_distance=moved_distance,
            lateral_error=lateral_error,
            old_coefficient=old_coefficient,
            new_coefficient=self.config.press_coefficient,
        )
        print(
            f"自动微调系数：{reason}，进度 {progress:.2f}，"
            f"{old_coefficient:.3f} -> {self.config.press_coefficient:.3f}"
        )
        return True

    def _blend_coefficient(self, target_coefficient: float) -> float:
        return (
            self.config.press_coefficient * (1 - self.config.coefficient_learning_rate)
            + target_coefficient * self.config.coefficient_learning_rate
        )

    def _record_coefficient_skip(
        self,
        reason: str,
        progress: float,
        planned_distance: float,
        distance_after: float,
        moved_distance: float,
        lateral_error: float | None,
    ) -> None:
        self.telemetry.record(
            "coefficient_adjust_skipped",
            reason=reason,
            progress=progress,
            planned_distance=planned_distance,
            distance_after=distance_after,
            moved_distance=moved_distance,
            lateral_error=lateral_error,
            coefficient=self.config.press_coefficient,
        )
