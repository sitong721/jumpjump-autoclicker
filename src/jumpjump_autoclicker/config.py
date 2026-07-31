from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for detection and jump control."""

    assets_dir: Path = PROJECT_ROOT / "assets" / "templates"
    debug_dir: Path = PROJECT_ROOT / "debug"
    telemetry_path: Path = PROJECT_ROOT / "debug" / "jump_history.jsonl"
    calibration_path: Path = PROJECT_ROOT / "debug" / "distance_calibration.json"
    telemetry_enabled: bool = True
    press_coefficient: float = 1.58
    debug_mode: bool = True
    game_ready_delay_seconds: float = 5.0
    start_delay_seconds: float = 3.0
    game_over_detection_enabled: bool = True
    game_over_button_min_width_ratio: float = 0.32
    game_over_button_min_height_ratio: float = 0.035
    game_over_button_max_height_ratio: float = 0.13
    game_over_button_min_y_ratio: float = 0.62
    game_over_button_max_y_ratio: float = 0.82
    game_over_panel_min_width_ratio: float = 0.38
    game_over_panel_min_height_ratio: float = 0.1
    game_over_overlay_min_ratio: float = 0.45
    game_over_min_signal_score: float = 2.6
    game_over_requires_replay_button: bool = True
    game_over_requires_overlay: bool = False
    ranking_detection_enabled: bool = True
    ranking_min_signals: int = 5
    ranking_dark_ratio_threshold: float = 0.62
    ranking_light_ratio_threshold: float = 0.25
    ranking_requires_dark_blocks: bool = True
    max_jumps: int = 0
    max_idle_rounds: int = 12
    max_failed_jumps: int = 0
    max_player_detection_fails: int = 6
    redetect_delay_seconds: float = 1.0
    background_match_threshold: float = 0.68
    background_match_scales: tuple[float, ...] = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)
    player_match_threshold: float = 0.45
    player_stable_match_threshold: float = 0.52
    player_low_confidence_confirmation_rounds: int = 2
    player_low_confidence_position_tolerance: float = 8.0
    player_color_fallback_enabled: bool = True
    player_color_min_area: int = 120
    player_color_max_area: int = 1600
    player_color_min_y_ratio: float = 0.32
    player_color_max_y_ratio: float = 0.95
    target_min_area: int = 180
    target_max_area_ratio: float = 0.35
    target_cluster_count: int = 5
    target_min_y_ratio: float = 0.3
    target_max_y_gap_ratio: float = 0.18
    target_search_below_player_ratio: float = 0.25
    target_top_y_ratio: float = 0.24
    target_top_band_ratio: float = 0.25
    below_player_target_penalty: float = 0.35
    below_player_close_target_max_y_ratio: float = 0.08
    player_exclusion_radius: float = 80.0
    current_platform_exclusion_radius: float = 115.0
    close_target_min_horizontal_separation: float = 130.0
    close_target_max_distance: float = 170.0
    close_target_score_boost: float = 1.45
    side_search_margin: int = 35
    vertical_distance_weight: float = 1.12
    wrong_side_penalty: float = 0.08
    fallback_target_min_area: int = 120
    success_distance_threshold: float = 50.0
    min_jump_distance: float = 140.0
    max_jump_distance: float = 520.0
    max_jump_distance_after_target_lost: float = 420.0
    far_target_confirmation_rounds: int = 2
    far_target_confirmation_tolerance: float = 35.0
    min_horizontal_separation: float = 45.0
    auto_adjust_coefficient: bool = True
    coefficient_adjust_step: float = 0.02
    coefficient_learning_rate: float = 0.28
    max_coefficient_change_ratio: float = 0.08
    min_adjust_moved_ratio: float = 0.35
    min_effective_moved_ratio: float = 0.18
    max_adjust_lateral_error_ratio: float = 0.22
    max_adjust_distance_after_ratio: float = 0.6
    next_jump_compensation_enabled: bool = False
    next_jump_compensation_gain: float = 0.2
    min_next_press_multiplier: float = 0.95
    max_next_press_multiplier: float = 1.05
    max_compensation_lateral_error_ratio: float = 0.18
    max_history_samples: int = 80
    distance_bucket_size: int = 10
    min_bucket_samples: int = 1
    bucket_learning_rate: float = 0.28
    max_interpolation_gap: float = 45.0
    min_bucket_coefficient: float = 1.25
    max_bucket_coefficient: float = 1.85
    min_initial_press_coefficient: float = 1.52
    max_initial_press_coefficient: float = 1.62
    max_history_restore_distance_after_ratio: float = 0.45
    min_valid_progress: float = 0.5
    max_valid_progress: float = 1.25
    max_press_time_ms: float = 900.0
    min_press_coefficient: float = 1.1
    max_press_coefficient: float = 1.72
    focus_game_before_jump: bool = True
    focus_click_y_offset: int = 22
    focus_delay_seconds: float = 0.25
