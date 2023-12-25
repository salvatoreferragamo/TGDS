from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

# @dataclass
# class AdaLoRAArguments:
#     """
#     一些自定义参数
#     """
#     max_seq_length: int = field(metadata={"help": "输入最大长度"})
#     train_file: str = field(metadata={"help": "训练集"})
#     model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
#     task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
#     eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
#     lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
#     lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
#     lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
#     adalora_init_r: Optional[int] = field(default=12, metadata={"help": "adalora param"})
#     adalora_tinit: Optional[int] = field(default=200, metadata={"help": "adalora param"})
#     adalora_tfinal: Optional[int] = field(default=1000, metadata={"help": "adalora param"})
#     adalora_delta_t: Optional[int] = field(default=10, metadata={"help": "adalora param"})