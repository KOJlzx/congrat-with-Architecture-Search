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

def main():
    """主训练函数"""
    # 获取环境变量
    worker_id = int(os.environ.get('WORKER_ID', 0))
    is_head = os.environ.get('IS_HEAD', 'False').lower() == 'true'
    head_address = os.environ.get('HEAD_ADDRESS', '127.0.0.1:6379')
    
    print(f"Worker {worker_id} starting... (is_head: {is_head})")
    
    # 使用绝对路径访问配置文件
    base_dir = "/home/neulab/lizaixi/congrat"
    config_path = os.path.join(base_dir, "configs/clip-graph/inductive-causal/pubmed/base.yaml")
    cfg = OmegaConf.load(config_path)
    seed_everything(cfg.seed_everything)

    # 手动实例化 DataModule 和 Model
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

    # 创建 Trainer，使用Ray的DDP策略和环境
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

    # 交给 Ray 接管 Trainer
    trainer = prepare_trainer(trainer)

    # 开始训练
    print(f"Worker {worker_id} starting training...")
    trainer.fit(model, datamodule=datamodule)

    # 保存 checkpoint
    with train.checkpoint_dir(step=trainer.current_epoch) as ckpt_dir:
        trainer.save_checkpoint(os.path.join(ckpt_dir, f"model_worker_{worker_id}.ckpt"))
    
    print(f"Worker {worker_id} training completed!")


if __name__ == "__main__":
    # 获取环境变量
    worker_id = int(os.environ.get('WORKER_ID', 0))
    is_head = os.environ.get('IS_HEAD', 'False').lower() == 'true'
    head_address = os.environ.get('HEAD_ADDRESS', '127.0.0.1:6379')
    
    try:
        if is_head:
            # Head worker: 启动Ray集群
            print(f"Starting Ray head node on {head_address}...")
            ray.init(
                address=None,  # 启动新的集群
                ignore_reinit_error=True,
                num_cpus=1,
                num_gpus=1,
                object_store_memory=1000000000,
                _node_ip_address=head_address.split(':')[0],
                _node_manager_port=int(head_address.split(':')[1]),
            )
            print(f"Ray head node started at {head_address}")
        else:
            # 普通worker: 连接到现有集群
            print(f"Connecting to Ray cluster at {head_address}...")
            ray.init(
                address=f"ray://{head_address}",
                ignore_reinit_error=True,
            )
            print(f"Connected to Ray cluster at {head_address}")
        
        # 运行主训练函数
        main()
        
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        raise
    finally:
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()
            print(f"Worker {worker_id} Ray resources cleaned up")
