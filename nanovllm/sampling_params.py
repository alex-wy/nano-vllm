from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0 # 标准softmax温度，温度越高，模型输出的随机性越强，为 0 时则为贪心，softmax概率接近100%
    max_tokens: int = 64 # 最大生成token数
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
