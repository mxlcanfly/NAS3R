from .loss import Loss
from .loss_child_grid import LossChildGrid, LossChildGridCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossChildGridCfgWrapper: LossChildGrid,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = (
    LossChildGridCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper
)

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
