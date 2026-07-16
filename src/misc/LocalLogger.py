from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, save_dir: Path = LOG_PATH) -> None:
        super().__init__()
        self._save_dir = Path(save_dir)
        self._experiment = SummaryWriter(self._save_dir / "tensorboard")

    @property
    def experiment(self):
        return self._experiment

    @property
    def save_dir(self):
        return str(self._save_dir)

    @property
    def log_dir(self):
        return str(self._save_dir / "tensorboard")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            if isinstance(value, (int, float, np.number)):
                self._experiment.add_scalar(key, value, step)

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self._save_dir / f"images/{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
            self._experiment.add_image(
                f"{key}/{index}", image, step, dataformats="HWC"
            )

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._experiment.flush()
        self._experiment.close()
