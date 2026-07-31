from __future__ import annotations

import argparse

from .app import JumpJumpApp


def main() -> None:
    parser = argparse.ArgumentParser(description="微信跳一跳自动化助手")
    parser.add_argument(
        "--verify-once",
        action="store_true",
        help="只截一帧并手动点选棋子/目标，用于排查识别偏差",
    )
    parser.add_argument(
        "--verify-during-run",
        action="store_true",
        help="自动跳跃过程中，每次跳前手动点选棋子/目标并打印识别偏差",
    )
    parser.add_argument(
        "--verify-limit",
        type=int,
        default=0,
        help="配合 --verify-during-run 使用，限制校验次数；0 表示不限制",
    )
    parser.add_argument(
        "--verify-resume-delay",
        type=float,
        default=5.0,
        help="配合 --verify-during-run 使用，手动点选完成后等待多少秒再继续跳跃",
    )
    parser.add_argument(
        "--step-check",
        action="store_true",
        help="step-by-step manual point check, then jump once with manual or auto points",
    )
    args = parser.parse_args()

    app = JumpJumpApp()
    if args.verify_once:
        app.run_manual_point_check()
        return
    if args.step_check:
        app.run_step_check()
        return

    app.run(
        verify_during_run=args.verify_during_run,
        verify_limit=args.verify_limit,
        verify_resume_delay=args.verify_resume_delay,
    )


if __name__ == "__main__":
    main()
