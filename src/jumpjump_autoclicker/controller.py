from __future__ import annotations

import time
from typing import TypeAlias

import cv2
import numpy as np
import pyautogui


Point: TypeAlias = tuple[int, int]
GameWindow: TypeAlias = tuple[int, int, int, int]


class DesktopController:
    """Screen capture and mouse operations for the desktop game window."""

    def __init__(self) -> None:
        self.screen_width, self.screen_height = pyautogui.size()
        self.game_window: GameWindow | None = None

    def set_game_window(self, window: GameWindow) -> None:
        self.game_window = window

    def capture_screen(self) -> np.ndarray:
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    def capture_game_screen(self) -> np.ndarray | None:
        if self.game_window is None:
            print("错误：未设置游戏区域")
            return None

        x, y, width, height = self.game_window
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    def focus_game_window(self, y_offset: int = 22, delay_seconds: float = 0.25) -> None:
        if self.game_window is None:
            return

        x, y, width, _ = self.game_window
        focus_x = x + width // 2
        focus_y = y + y_offset
        print(f"聚焦游戏窗口: ({focus_x}, {focus_y})")
        pyautogui.click(focus_x, focus_y)
        time.sleep(delay_seconds)

    def perform_jump(
        self,
        press_time_ms: float,
        player_pos: Point,
        focus_before_press: bool = False,
        focus_y_offset: int = 22,
        focus_delay_seconds: float = 0.25,
    ) -> None:
        if self.game_window is None:
            print("错误：未设置游戏区域")
            return

        if focus_before_press:
            self.focus_game_window(focus_y_offset, focus_delay_seconds)

        x, y, _, _ = self.game_window
        click_x = x + player_pos[0]
        click_y = y + player_pos[1]

        print(f"点击位置: ({click_x}, {click_y}), 按压时间: {press_time_ms:.1f}ms")
        pyautogui.moveTo(click_x, click_y, duration=0.1)
        time.sleep(0.1)
        pyautogui.mouseDown()
        time.sleep(press_time_ms / 1000)
        pyautogui.mouseUp()
        pyautogui.moveTo(click_x + 50, click_y, duration=0.1)
