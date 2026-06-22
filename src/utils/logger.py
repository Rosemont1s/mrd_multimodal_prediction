"""
Logging Utilities
==================
Unified logging interface supporting TensorBoard and Weights & Biases.
Also configures Python's standard logging for console and file output.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Logger:
    """Unified experiment logger with TensorBoard and W&B support.

    Provides a common interface for logging scalars, text, and configuration
    to TensorBoard, Weights & Biases, or both.  Also sets up Python's
    standard ``logging`` module with console and file handlers.

    Args:
        cfg: Configuration dictionary.  Expected keys::

            cfg['logging']['backend']       -> "tensorboard", "wandb", or "both"
            cfg['logging']['log_dir']       -> TensorBoard log directory
            cfg['logging']['wandb_project'] -> W&B project name
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        log_cfg = cfg.get("logging", {})
        self.backend = log_cfg.get("backend", "tensorboard")
        self.log_dir = log_cfg.get("log_dir", "runs/")

        # ---- TensorBoard ----
        self.tb_writer = None
        if self.backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard writer initialized: {self.log_dir}")
            except ImportError:
                logger.warning(
                    "TensorBoard not installed. Skipping TensorBoard logging."
                )

        # ---- Weights & Biases ----
        self.wandb_run = None
        if self.backend in ("wandb", "both"):
            try:
                import wandb
                project = log_cfg.get("wandb_project", "mrd_prediction")
                self.wandb_run = wandb.init(
                    project=project,
                    config=cfg,
                    reinit=True,
                )
                logger.info(f"W&B run initialized: project={project}")
            except ImportError:
                logger.warning(
                    "wandb not installed. Skipping W&B logging."
                )

        # ---- Python standard logging ----
        self._setup_python_logging()

    def _setup_python_logging(self) -> None:
        """Configure Python's standard logging with console and file handlers."""
        root_logger = logging.getLogger()

        # Only configure if no handlers exist yet
        if root_logger.handlers:
            return

        root_logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

        # File handler
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "training.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

        logger.info(f"Python logging configured. Log file: {log_file}")

    def log_scalars(self, tag_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.

        Args:
            tag_dict: Dictionary mapping tag names to scalar values.
            step: Global step number.
        """
        # TensorBoard
        if self.tb_writer is not None:
            for tag, value in tag_dict.items():
                self.tb_writer.add_scalar(tag, value, step)

        # W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log(tag_dict, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text content.

        Args:
            tag: Tag name.
            text: Text content to log.
            step: Global step number.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)

        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Log the full configuration.

        Args:
            cfg: Configuration dictionary to log.
        """
        import yaml
        config_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)

        if self.tb_writer is not None:
            self.tb_writer.add_text("config", f"```yaml\n{config_str}\n```", 0)

        if self.wandb_run is not None:
            import wandb
            wandb.config.update(cfg, allow_val_change=True)

        logger.info(f"Configuration logged:\n{config_str}")

    def close(self) -> None:
        """Close all logging backends and flush buffers."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            logger.info("TensorBoard writer closed.")

        if self.wandb_run is not None:
            import wandb
            wandb.finish()
            logger.info("W&B run finished.")
