import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Dict, Any, List, Optional
import numpy as np
import time
from dataclasses import dataclass

# 导入你的原始模块
from src.clip_graph.lit import LitClipGraph
from src.clip_graph.data.datamodule import PubmedGraphTextDataModule
from omegaconf import OmegaConf

@dataclass
class TrainingConfig:
    config_path: str
    num_workers: int = 4
    num_epochs: int = 10
    learning_rate: float = 3e-5
    sync_frequency: int = 1  # 每几个epoch同步一次
    checkpoint_frequency: int = 5  # 每几个epoch保存一次checkpoint
    log_frequency: int = 10  # 每几个batch打印一次日志

# 初始化Ray
ray.init(address="auto")

# 数据分片准备函数
@ray.remote
def prepare_graph_data_shard(rank: int, world_size: int, config_path: str):
    """准备图数据分片"""
    config = OmegaConf.load(config_path)
    
    # 创建数据模块
    datamodule = PubmedGraphTextDataModule(**config.data.init_args)
    datamodule.prepare_data()
    datamodule.setup()
    
    # 获取训练数据集
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    # 计算数据分片
    total_samples = len(train_dataset)
    samples_per_shard = total_samples // world_size
    start_idx = rank * samples_per_shard
    end_idx = start_idx + samples_per_shard if rank < world_size - 1 else total_samples
    
    # 创建分片索引
    indices = list(range(start_idx, end_idx))
    train_shard = torch.utils.data.Subset(train_dataset, indices)
    val_shard = torch.utils.data.Subset(val_dataset, indices)

    # 创建分片的数据加载器
    train_dataloader = DataLoader(
        train_shard,
        batch_size=config.data.init_args.batch_size,
        shuffle=True,
        num_workers=config.data.init_args.num_workers,
        pin_memory=config.data.init_args.pin_memory,
        collate_fn=train_dataset.__collate__
    )
    val_dataloader = DataLoader(
        val_shard,
        batch_size=config.data.init_args.batch_size,
        shuffle=False,
        num_workers=config.data.init_args.num_workers,
        pin_memory=config.data.init_args.pin_memory,
        collate_fn=val_dataset.__collate__
    )
    
    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'total_samples': total_samples,
        'samples_per_shard': samples_per_shard,
        'start_idx': start_idx,
        'end_idx': end_idx
    }


@ray.remote(num_gpus=1)
class AdvancedGraphModelTrainer:
    def __init__(self, config: TrainingConfig, worker_id: int, train_data_shard_ref=None, val_data_shard_ref=None):
        self.config = config
        self.worker_id = worker_id
        self.device = torch.device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.epoch = 0
        self.global_step = 0
        print("in_init", '_'*100)
        # 加载配置
        self.model_config = OmegaConf.load(config.config_path)
        
        # 初始化数据模块
        self._setup_data(train_data_shard_ref, val_data_shard_ref)
        
        # 初始化模型
        self._setup_model()
        
        # 初始化优化器
        self._setup_optimizer()
        
        print(f"Worker {self.worker_id} initialized on device {self.device}")

    def _setup_data(self, train_data_shard_ref=None, val_data_shard_ref=None):
        """设置数据模块"""
        print("in_setup_data", '_'*100)
        if train_data_shard_ref is not None and val_data_shard_ref is not None:
            # 使用预准备的数据分片
            data_info = ray.get(train_data_shard_ref)
            self.train_dataloader = data_info['train_dataloader']
            self.val_dataloader = data_info['val_dataloader']
            print(f"Worker {self.worker_id}: using pre-prepared train data shard "
                  f"({data_info['start_idx']}-{data_info['end_idx']})")
            print(f"Worker {self.worker_id}: using pre-prepared val data shard "
                  f"({data_info['start_idx']}-{data_info['end_idx']})")
        else:
            # 创建完整的数据模块（用于验证）
            data_args = self.model_config.data.init_args.copy()
            
            # 确保使用绝对路径
            data_dir = data_args['data_dir']
            if not os.path.isabs(data_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_dir = os.path.join(project_root, data_dir)
                data_args['data_dir'] = data_dir
            
            self.datamodule = PubmedGraphTextDataModule(**data_args)
            self.datamodule.prepare_data()
            self.datamodule.setup()
            self.train_dataloader = self.datamodule.train_dataloader()
        


    def _setup_model(self):
        """设置模型"""
        # 从配置中获取参数
        model_args = self.model_config.model.init_args.copy()
        
        # 只覆盖我们想要自定义的参数
        model_args['lr'] = self.config.learning_rate
        
        self.model = LitClipGraph(**model_args).to(self.device)

    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.model_config.model.init_args.weight_decay
        )

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """训练一个epoch"""
        self.model.train()
        self.epoch = epoch
        print("in_train_epoch")
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 将数据移动到设备
            print("in_train_epoch_batch")
            batch = self._move_batch_to_device(batch)
            print("in_train_epoch_batch_move_to_device")
            self.optimizer.zero_grad()
            print("in_train_epoch_batch_zero_grad")
            # 前向传播
            loss = self.model.training_step(batch, batch_idx)
            print("in_train_epoch_batch_forward_propagation")
            # 反向传播
            loss.backward()
            self.optimizer.step()
            print("in_train_epoch_batch_backward_propagation")
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 定期打印日志
            if batch_idx % self.config.log_frequency == 0:
                elapsed_time = time.time() - start_time
                print(f"Worker {self.worker_id}, Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "epoch_time": epoch_time,
            "global_step": self.global_step
        }

    def validate_epoch(self, epoch: int) -> Dict[str, Any]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                loss = self.model.validation_step(batch, batch_idx)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_time = time.time() - start_time
        
        return {
            "val_loss": avg_loss,
            "num_batches": num_batches,
            "val_time": val_time
        }

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将batch数据移动到设备"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch

    def get_model_state(self) -> Dict[str, Any]:
        """获取完整模型状态"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'worker_id': self.worker_id
        }

    def set_model_state(self, state_dict: Dict[str, Any]):
        """设置完整模型状态"""
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.epoch = state_dict.get('epoch', 0)
        self.global_step = state_dict.get('global_step', 0)

    def get_model_parameters(self) -> List[torch.Tensor]:
        """获取模型参数（用于参数平均）"""
        return [param.cpu().clone() for param in self.model.parameters()]

    def set_model_parameters(self, parameters: List[torch.Tensor]):
        """设置模型参数（用于参数平均）"""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data.copy_(new_param.to(self.device))

    def save_checkpoint(self, checkpoint_path: str):
        """保存checkpoint"""
        state_dict = self.get_model_state()
        torch.save(state_dict, checkpoint_path)
        print(f"Worker {self.worker_id} saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.set_model_state(state_dict)
        print(f"Worker {self.worker_id} loaded checkpoint from {checkpoint_path}")

# 参数平均函数
def average_parameters(parameter_lists: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """计算参数平均值"""
    if not parameter_lists:
        return []
    
    num_workers = len(parameter_lists)
    averaged_params = []
    
    for i in range(len(parameter_lists[0])):
        # 收集所有worker的第i个参数
        param_tensors = [params[i] for params in parameter_lists]
        
        # 计算平均值
        avg_param = torch.stack(param_tensors).mean(dim=0)
        averaged_params.append(avg_param)
    
    return averaged_params

def train_distributed_advanced(config: TrainingConfig):
    """高级分布式训练主函数"""
    
    # print(f"Starting advanced distributed training with {config.num_workers} workers")
    # print(f"Config: {config}")
    
    # 准备数据分片
    # print("Preparing data shards...")
    data_shard_futures = [
        prepare_graph_data_shard.remote(i, config.num_workers, config.config_path) 
        for i in range(config.num_workers)
    ]
    # 创建多个训练器，每个使用对应的数据分片
    trainers = [
        AdvancedGraphModelTrainer.remote(config, i, data_shard_futures[i]) 
        for i in range(config.num_workers)
    ]
    
    # 训练循环
    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*50}")
        
        # 并行训练
        train_futures = [trainer.train_epoch.remote(epoch) for trainer in trainers]
        train_results = ray.get(train_futures)
        
        # 并行验证
        val_futures = [trainer.validate_epoch.remote(epoch) for trainer in trainers]
        val_results = ray.get(val_futures)
        
        # 计算和打印统计信息
        train_losses = [r['loss'] for r in train_results]
        val_losses = [r['val_loss'] for r in val_results]
        epoch_times = [r['epoch_time'] for r in train_results]
        
        print(f"Training Results:")
        for i, (train_result, val_result) in enumerate(zip(train_results, val_results)):
            print(f"  Worker {i}: Train Loss: {train_result['loss']:.4f}, "
                  f"Val Loss: {val_result['val_loss']:.4f}, "
                  f"Time: {train_result['epoch_time']:.2f}s")
        
        print(f"Average Train Loss: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
        print(f"Average Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Average Epoch Time: {np.mean(epoch_times):.2f}s")
        
        # 参数同步
        if epoch % config.sync_frequency == 0:
            print("Synchronizing model parameters...")
            sync_start = time.time()
            
            # 获取所有模型的参数
            param_futures = [trainer.get_model_parameters.remote() for trainer in trainers]
            parameter_lists = ray.get(param_futures)
            
            # 计算参数平均值
            averaged_params = average_parameters(parameter_lists)
            
            # 更新所有模型
            update_futures = [trainer.set_model_parameters.remote(averaged_params) 
                            for trainer in trainers]
            ray.get(update_futures)
            
            sync_time = time.time() - sync_start
            print(f"Model parameters synchronized in {sync_time:.2f}s")
        
        # 保存checkpoint
        if epoch % config.checkpoint_frequency == 0:
            print("Saving checkpoints...")
            checkpoint_futures = [trainer.save_checkpoint.remote(f"checkpoint_epoch_{epoch}_worker_{i}.pth") 
                                for i, trainer in enumerate(trainers)]
            ray.get(checkpoint_futures)
    
    print("\nTraining completed!")
    
    # 保存最终模型
    final_state_futures = [trainer.get_model_state.remote() for trainer in trainers]
    final_states = ray.get(final_state_futures)
    
    # 保存第一个worker的模型作为最终模型
    torch.save(final_states[0], f"final_model_epoch_{config.num_epochs}.pth")
    print("Final model saved!")

# 运行高级分布式训练
if __name__ == "__main__":
    config = TrainingConfig(
        config_path="/home/neulab/lizaixi/congrat-copy/configs/clip-graph/inductive-causal/pubmed/base.yaml",
        num_workers=1,
        num_epochs=10,
        learning_rate=3e-5,
        sync_frequency=1,
        checkpoint_frequency=5,
        log_frequency=10
    )
    
    try:
        train_distributed_advanced(config)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()