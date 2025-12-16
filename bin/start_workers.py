import os
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.lightning import prepare_trainer
from ray.train.lightning import RayDDPStrategy
from ray.train.lightning import RayLightningEnvironment
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
import clip_graph as cg
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ray.train.torch import TorchTrainer

def train_func(config):
    # 使用绝对路径访问配置文件
    base_dir = "/home/neulab/lizaixi/congrat"
    config_path = config.get("config_path", os.path.join(base_dir, "configs/clip-graph/inductive-causal/pubmed/base.yaml"))
    cfg = OmegaConf.load(config_path)
    seed_everything(cfg.seed_everything)

    # 2. 手动实例化 DataModule 和 Model（避免Ray环境中的instantiate问题）
    from hydra.utils import get_class
    
    # 修改DataModule配置，使用绝对路径
    data_args = cfg.data.init_args.copy()
    data_args['data_dir'] = os.path.join(base_dir, data_args['data_dir'])
    
    # 实例化DataModule
    datamodule_class = get_class(cfg.data.class_path)
    datamodule = datamodule_class(**data_args)
    
    # 实例化Model
    model_class = get_class(cfg.model.class_path)
    model = model_class(**cfg.model.init_args)

    # 3. 手动创建 Trainer，使用Ray的DDP策略和环境
    trainer = Trainer(
        enable_checkpointing=True,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        precision=16,
        log_every_n_steps=10,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False,
        detect_anomaly=False,
        max_epochs=20,
        min_epochs=2,
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
    )

    # 4. 交给 Ray 接管 Trainer
    trainer = prepare_trainer(trainer)

    # 5. 开始训练
    trainer.fit(model, datamodule=datamodule)

    # 6. 保存 checkpoint
    with train.checkpoint_dir(step=trainer.current_epoch) as ckpt_dir:
        trainer.save_checkpoint(os.path.join(ckpt_dir, "model.ckpt"))


if __name__ == "__main__":
    ray.init(address="auto")  # 连接 Ray 集群

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=2,   # ⚡ worker 数量 = 节点数 × 每节点 GPU 数
            use_gpu=True,
        ),
        run_config=RunConfig(
            name="clip_graph_ray",
            storage_path=os.path.abspath("./ay_results"),
        ),
    )

    result = trainer.fit()
    print("Final:", result)
