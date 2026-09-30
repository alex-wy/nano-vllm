import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True) # dataclass，类装饰器，自动生成init,repr,eq基础方法用于初始化，打印，比较； slot 主要为了去掉 __dict__ 字典的开销
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 单步前向 批内 token 总数 上限，应该不小于单条最长序列长度
    max_num_seqs: int = 512 # 并发序列条数上限
    max_model_len: int = 4096 # 单序列最大上下文长度
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False # True 时为eager模式，不用 CUDA Graph，便于调试
    hf_config: AutoConfig | None = None
    eos: int = -1 # 结束符 token id，默认 -1 表示后续再设
    kvcache_block_size: int = 256 # KV分页大小，PagedAttention 思想
    num_kvcache_blocks: int = -1 # -1 代表自动算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0 # 需 256 的倍数（与内核/对齐有关）
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 调度可行性：至少能 单步吞下一条满长 prompt 的 prefill
