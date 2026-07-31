from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .assets import TemplateImage
from .config import AppConfig
from .controller import GameWindow, Point
from .debug import DebugWriter


@dataclass(slots=True)
class TargetCandidate:
    area: float
    landing_point: Point
    bbox: tuple[int, int, int, int]
    cluster_id: int
    score: float
    source: str = "kmeans"


class VisionDetector:
    """OpenCV based game-window, player, and target detection."""

    def __init__(
        self,
        config: AppConfig,
        debug_writer: DebugWriter,
        player_template: TemplateImage | None,
        background_templates: list[TemplateImage],
    ) -> None:
        self.config = config
        self.debug_writer = debug_writer
        self.player_template = player_template
        self.background_templates = background_templates
        self.last_player_match_score: float | None = None
        self.last_player_detection_source: str | None = None

    def auto_detect_game_window(self, screenshot: np.ndarray) -> GameWindow | None:
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        best_score = -1.0
        best_window: GameWindow | None = None
        best_template = "无"
        screenshot_h, screenshot_w = gray_screenshot.shape[:2]

        for template in self.background_templates:
            gray_template = cv2.cvtColor(template.image, cv2.COLOR_BGR2GRAY)
            template_h, template_w = gray_template.shape[:2]

            for scale in self.config.background_match_scales:
                scaled_w = int(template_w * scale)
                scaled_h = int(template_h * scale)
                if scaled_w <= 0 or scaled_h <= 0:
                    continue
                if scaled_w > screenshot_w or scaled_h > screenshot_h:
                    continue

                if scale == 1.0:
                    candidate_template = gray_template
                else:
                    candidate_template = cv2.resize(gray_template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

                result = cv2.matchTemplate(gray_screenshot, candidate_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = float(max_val)
                    best_window = (max_loc[0], max_loc[1], scaled_w, scaled_h)
                    best_template = f"{template.path.name if template.path else 'unknown'} x{scale:.2f}"

        print(f"背景模板最佳匹配: {best_score:.3f} ({best_template})")
        if best_window is not None and best_score > self.config.background_match_threshold:
            return best_window

        return None

    def find_player_position(self, image: np.ndarray) -> Point | None:
        self.last_player_match_score = None
        self.last_player_detection_source = None
        if self.player_template is None:
            print("警告：未找到棋子模板")
            return None

        result = cv2.matchTemplate(
            image,
            self.player_template.image,
            cv2.TM_CCOEFF_NORMED,
            mask=self.player_template.mask,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.last_player_match_score = float(max_val) if np.isfinite(max_val) else None
        print(f"棋子模板匹配值: {max_val:.3f}")

        if not np.isfinite(max_val) or max_val <= self.config.player_stable_match_threshold:
            fallback_pos = self._find_player_by_color(image)
            if fallback_pos is not None:
                self.last_player_detection_source = "color"
                print(f"棋子颜色兜底位置: {fallback_pos}")
                return fallback_pos
            if np.isfinite(max_val) and max_val > self.config.player_match_threshold:
                self.last_player_detection_source = "template"
                return self._player_position_from_template(image, max_loc)
            return None

        self.last_player_detection_source = "template"
        return self._player_position_from_template(image, max_loc)

    def _player_position_from_template(self, image: np.ndarray, max_loc: Point) -> Point:
        template_h, template_w = self.player_template.image.shape[:2]
        player_x = max_loc[0] + template_w // 2
        player_y = max_loc[1] + template_h - template_h // 4

        debug_img = image.copy()
        cv2.rectangle(debug_img, max_loc, (max_loc[0] + template_w, max_loc[1] + template_h), (0, 255, 0), 2)
        cv2.circle(debug_img, (player_x, player_y), 5, (0, 0, 255), -1)
        self.debug_writer.save("player_template", debug_img)

        return (player_x, player_y)

    def _find_player_by_color(self, image: np.ndarray) -> Point | None:
        if not self.config.player_color_fallback_enabled:
            return None

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        purple_mask = cv2.inRange(hsv, (105, 35, 25), (150, 255, 205))
        bgr = image
        blue_channel = bgr[:, :, 0]
        green_channel = bgr[:, :, 1]
        red_channel = bgr[:, :, 2]
        purple_bgr_mask = (
            (blue_channel > 45)
            & (green_channel < 115)
            & (red_channel < 145)
            & (blue_channel > green_channel + 8)
        ).astype(np.uint8) * 255
        purple_mask = cv2.bitwise_or(purple_mask, purple_bgr_mask)

        y_min = int(height * self.config.player_color_min_y_ratio)
        y_max = int(height * self.config.player_color_max_y_ratio)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y_min:y_max, :] = 255
        purple_mask = cv2.bitwise_and(purple_mask, roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        purple_mask = cv2.dilate(purple_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, Point, tuple[int, int, int, int]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.player_color_min_area or area > self.config.player_color_max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect = w / h
            if aspect < 0.28 or aspect > 0.95:
                continue
            if h < height * 0.045 or h > height * 0.13:
                continue
            if y + h < height * self.config.player_color_min_y_ratio:
                continue

            player_x = x + w // 2
            player_y = y + h
            center_bias = 1.0 - min(0.5, abs(player_x - width / 2) / width)
            shape_score = 1.0 - min(0.6, abs(aspect - 0.55))
            score = area * center_bias * shape_score
            candidates.append((score, (player_x, player_y), (x, y, w, h)))

        if not candidates:
            return None

        _, player_pos, bbox = max(candidates, key=lambda item: item[0])
        debug_img = image.copy()
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.circle(debug_img, player_pos, 5, (0, 0, 255), -1)
        self.debug_writer.save("player_color_fallback", debug_img)
        return player_pos

    def is_game_over_screen(self, image: np.ndarray) -> bool:
        if not self.config.game_over_detection_enabled:
            return False

        if self.is_ranking_screen(image):
            return True

        height, width = image.shape[:2]
        signals = {
            "button": self._has_game_over_replay_button(image, width, height),
            "score": self._has_game_over_score_panel(image, width, height),
            "overlay": self._has_game_over_gray_overlay(image, width, height),
            "bottom_actions": self._has_game_over_bottom_actions(image, width, height),
        }
        signal_score = (
            (1.5 if signals["button"] else 0.0)
            + (1.0 if signals["score"] else 0.0)
            + (0.7 if signals["overlay"] else 0.0)
            + (0.7 if signals["bottom_actions"] else 0.0)
        )

        print(
            "结算页检测: "
            + ", ".join(f"{key}={value}" for key, value in signals.items())
            + f", score={signal_score:.1f}"
        )

        is_game_over = signal_score >= self.config.game_over_min_signal_score
        if self.config.game_over_requires_replay_button:
            is_game_over = is_game_over and signals["button"]
        if self.config.game_over_requires_overlay:
            is_game_over = is_game_over and signals["overlay"]

        if is_game_over:
            debug_img = image.copy()
            cv2.putText(
                debug_img,
                "Game over detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            self.debug_writer.save("game_over", debug_img)
            return True

        return False

    def is_ranking_screen(self, image: np.ndarray) -> bool:
        if not self.config.ranking_detection_enabled:
            return False

        height, width = image.shape[:2]
        signals = {
            "title": self._ranking_title_band_signal(image, width, height),
            "main_panel": self._ranking_main_panel_signal(image, width, height),
            "bottom_row": self._ranking_bottom_row_signal(image, width, height),
            "back_button": self._ranking_back_button_signal(image, width, height),
            "group_button": self._ranking_group_button_signal(image, width, height),
        }
        signal_count = sum(signals.values())
        print("排行榜检测: " + ", ".join(f"{key}={value}" for key, value in signals.items()))

        has_required_blocks = signals["main_panel"] and signals["bottom_row"]
        if self.config.ranking_requires_dark_blocks and not has_required_blocks:
            return False
        if not signals["title"]:
            return False

        if signal_count >= self.config.ranking_min_signals:
            debug_img = image.copy()
            self._draw_ranking_regions(debug_img, width, height)
            cv2.putText(
                debug_img,
                "Ranking screen detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            self.debug_writer.save("ranking_screen", debug_img)
            return True

        return False

    def _ranking_title_band_signal(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.12) : int(height * 0.22), int(width * 0.25) : int(width * 0.75)]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        return cv2.countNonZero(white) / max(1, white.size) > 0.04

    def _ranking_main_panel_signal(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.22) : int(height * 0.72), int(width * 0.08) : int(width * 0.92)]
        return self._dark_region_ratio(region) >= self.config.ranking_dark_ratio_threshold

    def _ranking_bottom_row_signal(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.73) : int(height * 0.86), int(width * 0.08) : int(width * 0.92)]
        return self._dark_region_ratio(region) >= self.config.ranking_dark_ratio_threshold

    def _ranking_back_button_signal(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.86) : int(height * 0.98), int(width * 0.03) : int(width * 0.2)]
        return self._light_region_ratio(region) >= self.config.ranking_light_ratio_threshold

    def _ranking_group_button_signal(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.86) : int(height * 0.98), int(width * 0.52) : int(width * 0.94)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_ratio = cv2.countNonZero(edges) / max(1, edges.size)
        return edge_ratio > 0.025 or self._light_region_ratio(region) > 0.12

    @staticmethod
    def _dark_region_ratio(region: np.ndarray) -> float:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(hsv, (0, 0, 20), (180, 90, 105))
        return cv2.countNonZero(dark) / max(1, dark.size)

    @staticmethod
    def _light_region_ratio(region: np.ndarray) -> float:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        light = cv2.inRange(hsv, (0, 0, 190), (180, 90, 255))
        return cv2.countNonZero(light) / max(1, light.size)

    @staticmethod
    def _draw_ranking_regions(image: np.ndarray, width: int, height: int) -> None:
        regions = [
            (0.25, 0.12, 0.75, 0.22),
            (0.08, 0.22, 0.92, 0.72),
            (0.08, 0.73, 0.92, 0.86),
            (0.03, 0.86, 0.2, 0.98),
            (0.52, 0.86, 0.94, 0.98),
        ]
        for left, top, right, bottom in regions:
            cv2.rectangle(
                image,
                (int(width * left), int(height * top)),
                (int(width * right), int(height * bottom)),
                (0, 0, 255),
                2,
            )

    def _has_game_over_replay_button(self, image: np.ndarray, width: int, height: int) -> bool:
        lower_y = int(height * self.config.game_over_button_min_y_ratio)
        lower_bottom = int(height * self.config.game_over_button_max_y_ratio)
        lower = image[lower_y:lower_bottom, :]
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 205), (180, 70, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, button_width, button_height = cv2.boundingRect(contour)
            absolute_y = y + lower_y
            center_x = x + button_width / 2
            center_y = absolute_y + button_height / 2
            if button_width < width * self.config.game_over_button_min_width_ratio:
                continue
            if button_height < height * self.config.game_over_button_min_height_ratio:
                continue
            if button_height > height * self.config.game_over_button_max_height_ratio:
                continue
            if center_y < height * self.config.game_over_button_min_y_ratio:
                continue
            if center_y > height * self.config.game_over_button_max_y_ratio:
                continue
            if center_x < width * 0.2 or center_x > width * 0.8:
                continue
            return True

        return False

    def _has_game_over_score_panel(self, image: np.ndarray, width: int, height: int) -> bool:
        score_region = image[int(height * 0.14) : int(height * 0.42), int(width * 0.2) : int(width * 0.8)]
        score_hsv = cv2.cvtColor(score_region, cv2.COLOR_BGR2HSV)
        score_white = cv2.inRange(score_hsv, (0, 0, 185), (180, 75, 255))
        white_ratio = cv2.countNonZero(score_white) / max(1, score_white.size)
        score_gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(score_gray, 40, 120)
        edge_ratio = cv2.countNonZero(edges) / max(1, edges.size)
        if white_ratio >= 0.025 and edge_ratio >= 0.01:
            return True

        middle_y = int(height * 0.24)
        middle = image[middle_y : int(height * 0.76), :]
        hsv = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, (0, 0, 25), (180, 90, 145))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            _, _, panel_width, panel_height = cv2.boundingRect(contour)
            if panel_width < width * self.config.game_over_panel_min_width_ratio:
                continue
            if panel_height < height * self.config.game_over_panel_min_height_ratio:
                continue
            return True

        return False

    def _has_game_over_gray_overlay(self, image: np.ndarray, width: int, height: int) -> bool:
        center = image[int(height * 0.12) : int(height * 0.9), int(width * 0.08) : int(width * 0.92)]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        grayish = cv2.inRange(hsv, (0, 0, 90), (180, 45, 220))
        ratio = cv2.countNonZero(grayish) / max(1, grayish.size)
        return ratio >= self.config.game_over_overlay_min_ratio

    def _has_game_over_bottom_actions(self, image: np.ndarray, width: int, height: int) -> bool:
        region = image[int(height * 0.64) : int(height * 0.86), int(width * 0.08) : int(width * 0.92)]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 80, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        wide_actions = 0
        round_actions = 0
        for contour in contours:
            x, _, action_width, action_height = cv2.boundingRect(contour)
            if action_width < width * 0.08 or action_height < height * 0.035:
                continue
            if action_height > height * 0.14:
                continue

            aspect = action_width / max(1, action_height)
            center_x = x + action_width / 2
            if 1.8 <= aspect <= 7.5 and width * 0.2 <= center_x <= width * 0.8:
                wide_actions += 1
            elif 0.65 <= aspect <= 1.45:
                round_actions += 1

        return wide_actions >= 1 and round_actions >= 1

    def find_target_position(self, image: np.ndarray, player_pos: Point | None) -> Point | None:
        if player_pos is None:
            return None

        _player_x, player_y = player_pos
        image_height, _image_width = image.shape[:2]
        search_bottom = min(image_height, player_y + int(image_height * self.config.target_search_below_player_ratio))
        search_area = image[0:search_bottom, :]
        if search_area.size == 0:
            return None

        self.debug_writer.save("search_area", search_area)
        search_mask = self._target_search_mask(search_area.shape[:2], image.shape[:2], player_pos)

        data = np.float32(search_area.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(
            data,
            self.config.target_cluster_count,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        labels = labels.reshape(search_area.shape[:2])

        candidates = self._target_candidates(labels, image.shape[:2], player_pos, search_mask)
        print(f"目标候选(K-means): {len(candidates)}")
        fallback_candidates = self._fallback_target_candidates(search_area, image.shape[:2], player_pos, search_mask)
        print(f"目标候选(边缘兜底): {len(fallback_candidates)}")
        candidates.extend(fallback_candidates)
        if not candidates:
            self._save_no_target_debug(image, player_pos)
            return None

        candidates.sort(key=lambda item: item.score, reverse=True)
        best = candidates[0]
        distance = self.weighted_distance(player_pos, best.landing_point)
        self._save_target_debug(image, player_pos, best, distance, candidates)
        return best.landing_point

    def weighted_distance(self, start: Point, end: Point) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return float(np.sqrt(dx * dx + dy * dy * self.config.vertical_distance_weight))

    def _target_candidates(
        self,
        labels: np.ndarray,
        image_size: tuple[int, int],
        player_pos: Point,
        search_mask: np.ndarray,
    ) -> list[TargetCandidate]:
        candidates: list[TargetCandidate] = []
        image_height, image_width = image_size
        image_area = image_height * image_width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for cluster_id in range(self.config.target_cluster_count):
            cluster_mask = (labels == cluster_id).astype(np.uint8) * 255
            cluster_mask = cv2.bitwise_and(cluster_mask, search_mask)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
            edges = cv2.Canny(cluster_mask, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.target_min_area:
                    continue

                bbox = cv2.boundingRect(contour)
                if not self._is_reasonable_target_bbox(bbox, image_width, image_area):
                    continue

                landing_point = self._landing_point(contour, bbox)
                if landing_point is None:
                    continue

                score = self._candidate_score(area, landing_point, bbox, image_size, player_pos)
                if score <= 0:
                    continue

                candidates.append(TargetCandidate(area, landing_point, bbox, cluster_id, score, "kmeans"))

        return candidates

    def _fallback_target_candidates(
        self,
        search_area: np.ndarray,
        image_size: tuple[int, int],
        player_pos: Point,
        search_mask: np.ndarray,
    ) -> list[TargetCandidate]:
        candidates: list[TargetCandidate] = []
        image_height, image_width = image_size
        image_area = image_height * image_width

        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 35, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.bitwise_and(edges, search_mask)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.fallback_target_min_area:
                continue

            bbox = cv2.boundingRect(contour)
            if not self._is_reasonable_target_bbox(bbox, image_width, image_area, relaxed=True):
                continue

            landing_point = self._landing_point(contour, bbox)
            if landing_point is None:
                continue

            score = self._candidate_score(area, landing_point, bbox, image_size, player_pos, relaxed=True)
            if score <= 0:
                continue

            candidates.append(TargetCandidate(area, landing_point, bbox, -1, score, "edge"))

        return candidates

    def _target_search_mask(
        self,
        search_shape: tuple[int, int],
        image_size: tuple[int, int],
        player_pos: Point,
    ) -> np.ndarray:
        search_height, search_width = search_shape
        image_height, image_width = image_size
        player_x, player_y = player_pos
        mask = np.zeros((search_height, search_width), dtype=np.uint8)

        if player_x > image_width / 2:
            x_start = 0
            x_end = max(0, player_x - self.config.side_search_margin)
        else:
            x_start = min(image_width, player_x + self.config.side_search_margin)
            x_end = image_width

        y_start = int(image_height * self.config.target_min_y_ratio)
        y_end = min(search_height, player_y + int(image_height * self.config.target_search_below_player_ratio))
        if x_end > x_start and y_end > y_start:
            mask[y_start:y_end, x_start:x_end] = 255

        cv2.circle(mask, player_pos, int(self.config.current_platform_exclusion_radius), 0, -1)
        return mask

    def _is_reasonable_target_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_area: int,
        relaxed: bool = False,
    ) -> bool:
        x, _, width, height = bbox
        edge_margin = 1 if relaxed else 2
        min_width = 14 if relaxed else 20
        min_height = 8 if relaxed else 12
        max_area_ratio = self.config.target_max_area_ratio * (1.35 if relaxed else 1.0)

        if x <= edge_margin or x + width >= image_width - edge_margin:
            return False
        if width < min_width or height < min_height:
            return False
        if width * height > image_area * max_area_ratio:
            return False
        return True

    def _landing_point(self, contour: np.ndarray, bbox: tuple[int, int, int, int]) -> Point | None:
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            return None

        points = contour.reshape(-1, 2)
        top_y = int(points[:, 1].min())
        top_band_height = max(6, int(height * self.config.target_top_band_ratio))
        top_points = points[points[:, 1] <= top_y + top_band_height]

        if len(top_points) >= 2:
            landing_x = int(np.median(top_points[:, 0]))
        else:
            landing_x = x + width // 2

        landing_y = y + int(height * self.config.target_top_y_ratio)
        return (landing_x, landing_y)

    def _candidate_score(
        self,
        area: float,
        landing_point: Point,
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        player_pos: Point,
        relaxed: bool = False,
    ) -> float:
        image_height, image_width = image_size
        target_x, target_y = landing_point
        player_x, player_y = player_pos

        min_y_ratio = self.config.target_min_y_ratio * (0.75 if relaxed else 1.0)
        if target_y < image_height * min_y_ratio:
            return 0

        distance = self.weighted_distance(player_pos, landing_point)
        horizontal_separation = abs(target_x - player_x)
        is_close_target = (
            distance <= self.config.close_target_max_distance
            and horizontal_separation >= self.config.close_target_min_horizontal_separation
        )
        exclusion_radius = self.config.player_exclusion_radius * (0.75 if relaxed else 1.0)
        if distance < exclusion_radius and not is_close_target:
            return 0

        below_player_factor = 1.0
        below_player_gap = target_y - player_y
        if below_player_gap > 0:
            max_close_gap = image_height * self.config.below_player_close_target_max_y_ratio
            if not is_close_target or below_player_gap > max_close_gap:
                return 0
            below_player_factor = self.config.below_player_target_penalty

        if target_y >= player_y + image_height * self.config.target_search_below_player_ratio:
            return 0

        bbox_x, _, bbox_width, bbox_height = bbox
        shape_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
        min_shape_ratio = 0.12 if relaxed else 0.18
        if shape_ratio < min_shape_ratio:
            return 0

        preferred_side = -1 if player_x > image_width / 2 else 1
        actual_side = -1 if target_x < player_x else 1
        if actual_side != preferred_side:
            return 0
        side_factor = 1.0

        horizontal_factor = min(1.0, horizontal_separation / max(1, image_width * 0.35))
        vertical_separation = min(1.0, abs(target_y - player_y) / max(1, image_height * 0.35))
        slope_factor = self._target_slope_factor(player_pos, landing_point, is_close_target)
        if slope_factor <= 0:
            return 0
        size_factor = min(1.0, area / max(1, self.config.target_min_area * 8))
        distance_factor = 0.65 + 0.35 * min(1.0, distance / max(1, self.config.min_jump_distance))
        close_boost = self.config.close_target_score_boost if is_close_target else 1.0

        # Favor plausible next platforms over big background patches.
        center_bias = 1.0 - min(0.4, abs((bbox_x + bbox_width / 2) - image_width / 2) / image_width)
        return (
            area
            * side_factor
            * (0.5 + horizontal_factor)
            * (0.6 + vertical_separation)
            * size_factor
            * distance_factor
            * close_boost
            * below_player_factor
            * slope_factor
            * center_bias
        )

    def _target_slope_factor(self, player_pos: Point, landing_point: Point, is_close_target: bool) -> float:
        player_x, player_y = player_pos
        target_x, target_y = landing_point
        horizontal_gap = abs(target_x - player_x)
        vertical_gap = player_y - target_y

        if horizontal_gap <= 0 or vertical_gap <= 0:
            return 1.0 if is_close_target else 0.0

        slope = vertical_gap / horizontal_gap
        min_slope = self.config.target_min_slope_ratio
        ideal_slope = self.config.target_ideal_slope_ratio
        max_slope = self.config.target_max_slope_ratio

        if slope < min_slope and not is_close_target:
            return 0.0
        if slope < ideal_slope:
            return max(0.25, (slope / ideal_slope) ** 2)
        if slope > max_slope:
            return max(0.35, max_slope / slope)
        return 1.0

    def _save_no_target_debug(self, image: np.ndarray, player_pos: Point) -> None:
        debug_img = image.copy()
        cv2.circle(debug_img, player_pos, 7, (0, 0, 255), -1)
        cv2.circle(debug_img, player_pos, int(self.config.player_exclusion_radius), (0, 0, 255), 1)
        cv2.putText(
            debug_img,
            "No target candidates",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        self.debug_writer.save("target_not_found", debug_img)

    def _save_target_debug(
        self,
        image: np.ndarray,
        player_pos: Point,
        target: TargetCandidate,
        distance: float,
        candidates: list[TargetCandidate] | None = None,
    ) -> None:
        debug_img = image.copy()
        for index, candidate in enumerate((candidates or [])[:8], start=1):
            bx, by, bw, bh = candidate.bbox
            color = (0, 128, 255) if candidate.source == "edge" else (255, 128, 0)
            cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.circle(debug_img, candidate.landing_point, 4, color, -1)
            cv2.putText(
                debug_img,
                f"{index}:{candidate.score:.0f}",
                (candidate.landing_point[0] + 6, candidate.landing_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        bx, by, bw, bh = target.bbox
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.circle(debug_img, target.landing_point, 6, (255, 0, 0), -1)
        cv2.line(debug_img, player_pos, target.landing_point, (0, 255, 255), 2)
        cv2.putText(
            debug_img,
            f"Distance: {distance:.1f} Score: {target.score:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        self.debug_writer.save("target_shape", debug_img)
