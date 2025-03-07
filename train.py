import os

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@hydra.main(version_base=None, config_path="conf", config_name="config") # 允许你通过命令行参数覆盖配置文件中的值，会解析命令行参数，并将这些参数与配置文件中的内容合并。
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir # 返回当前运行时的输出目录。

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="Forecast-MAE",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1), # 提供模型结构的摘要信息，使用 rich 库进行美化输出。max_depth=1: 只显示模型的第一层结构，适用于大型模型时减少输出信息量
        RichProgressBar(), #替换默认的进度条为更美观、信息更丰富的进度条
        LearningRateMonitor(logging_interval="epoch"),# 监控并记录学习率的变化，方便后续分析和调试；logging_interval="epoch": 每个 epoch 结束时记录一次学习率。也可以设置为 "step" 来记录每个 batch 的学习率变化
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    # instantiate函数根据配置文件中的定义创建相应的对象
    model = instantiate(conf.model.target)
    datamodule = instantiate(conf.datamodule)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)


if __name__ == "__main__":
    main()
