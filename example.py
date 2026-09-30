import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        # 如果不使用 apply_chat_template，直接把文本丢给模型，模型可能无法正确区分哪里是用户说的话、哪里该由它来回答。通过这个函数，框架能自动适配当前模型（比如 Qwen3）对应的官方提示词格式，确保模型的输出质量和对齐效果。
        tokenizer.apply_chat_template( # 生成 带角色标记 的字符串，便于指令模型理解 user/assistant 边界。
            [{"role": "user", "content": prompt}],
            tokenize=False, # 设置为 False 表示只做字符串格式化，不直接转成数字 ID（Token IDs）。
            add_generation_prompt=True, # 在字符串的最后面加上模型专属的助手前缀
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params) # 核心入口：内部 tokenize + 调度 schedule + 前向 forward + 采样 sample

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
