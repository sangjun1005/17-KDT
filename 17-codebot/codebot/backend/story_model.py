import importlib.util


def load_storybot(directory, device):
    # Load the user's original files without conflicting with codebot's module names.
    modules = {}
    for name in ("model", "tokenizer"):
        spec = importlib.util.spec_from_file_location(f"original_storybot_{name}", directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    model = modules["model"].GPT.load_from(directory / "model_pretrain.pt", device=device)
    tokenizer = modules["tokenizer"].BPETokenizer.load_from(directory / "merge_rules.pkl")
    if model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("스토리봇 모델과 토크나이저의 어휘 크기가 일치하지 않습니다.")
    model.eval()
    return model, tokenizer
