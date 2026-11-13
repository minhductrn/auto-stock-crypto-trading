# Manual Session Observation

From the repository root, select a completed NYSE session and an explicit model directory.

    MODEL_DIR=artifacts/models/sdf82_shadow_20261113 python -m api.observe_session 2026-11-12

Add --fetch to retrieve the selected session from yfinance. Without that flag the command replays local daily bars. The local historical data supplies the universe and feature history. Each run replaces the local observation snapshot. Missing or invalid session bars produce an Incomplete snapshot and a nonzero exit code.

Run the frontend from its own directory and open the preview with ?mode=observation. This mode reads the snapshot and disables order submission. The snapshot is ignored by Git. The browser displays the update time in its local timezone.

The production schedule settings in configs/eod_schedule.json are disabled. No scheduler is installed or started by this workflow.
