"""Models the behavioural tests can call live. All share one method: classify(sentences, prompt) -> raw outputs.

- Baseline: TF-IDF + logistic regression trained on a split's training data (no prompt; ignores it)
- Ollama:  a local model (no API key; nothing leaves the machine)
- OpenAI:  any OpenAI-compatible chat endpoint (base_url + model; key from the environment)
- MLX:     an MLX model with an optional LoRA adapter, e.g. the fed-tone-finetune adapter (Apple Silicon only)
"""
import json
import os
import urllib.request

PROMPTS = {
    "original": (
        "Classify the monetary-policy stance of this central bank sentence.\n"
        "hawkish = points to tighter policy (higher rates, inflation worries).\n"
        "dovish = points to easier policy (lower rates, weak growth or jobs).\n"
        "neutral = neither.\n"
        "Answer with one word: hawkish, dovish or neutral.\n\nSentence: {sentence}"),
    "reordered": (
        "Read the central bank sentence and give its policy stance.\n"
        "neutral = no policy direction.\n"
        "dovish = easier policy (rate cuts, weak activity).\n"
        "hawkish = tighter policy (rate hikes, inflation concern).\n"
        "Reply with exactly one word: neutral, dovish or hawkish.\n\nSentence: {sentence}"),
    "terse": "Is this FOMC sentence hawkish, dovish or neutral? One word.\n\n{sentence}",
}


class Baseline:
    name = "TF-IDF + logistic regression (benchmark)"
    uses_prompt = False
    deterministic = True

    def __init__(self, train_df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(train_df.sentence)
        self.clf = LogisticRegression(max_iter=3000, C=4, class_weight="balanced").fit(X, train_df.label)

    def classify(self, sentences, prompt="original"):
        return list(self.clf.predict(self.vec.transform(list(sentences))))


class Ollama:
    uses_prompt = True
    deterministic = False

    def __init__(self, model, url="http://localhost:11434", temperature=0.0):
        self.model, self.url, self.temperature = model, url.rstrip("/"), temperature
        self.name = f"ollama:{model}"

    def _one(self, text):
        body = json.dumps({"model": self.model, "prompt": text, "stream": False,
                           "options": {"temperature": self.temperature, "num_predict": 8}}).encode()
        req = urllib.request.Request(self.url + "/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]

    def classify(self, sentences, prompt="original"):
        return [self._one(PROMPTS[prompt].format(sentence=s)) for s in sentences]


class OpenAICompatible:
    uses_prompt = True
    deterministic = False

    def __init__(self, model, base_url, api_key_env="OPENAI_API_KEY", temperature=0.0):
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self.key = os.environ.get(api_key_env, "")
        self.name = f"openai-compatible:{model}"

    def _one(self, text):
        body = json.dumps({"model": self.model, "temperature": self.temperature, "max_tokens": 8,
                           "messages": [{"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def classify(self, sentences, prompt="original"):
        return [self._one(PROMPTS[prompt].format(sentence=s)) for s in sentences]


class MLX:
    uses_prompt = True
    deterministic = True  # greedy decoding

    def __init__(self, model, adapter=None):
        from mlx_lm import load  # Apple Silicon only
        self.model, self.tok = load(model, adapter_path=adapter)
        self.name = f"mlx:{model}" + (f"+{os.path.basename(adapter)}" if adapter else "")

    def classify(self, sentences, prompt="original"):
        from mlx_lm import generate
        out = []
        for s in sentences:
            msgs = [{"role": "user", "content": PROMPTS[prompt].format(sentence=s)}]
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            out.append(generate(self.model, self.tok, prompt=p, max_tokens=8, verbose=False))
        return out


def make(cfg, train_df):
    b = cfg.get("live", {}) or {}
    kind = b.get("backend", "baseline")
    if kind == "baseline":
        return Baseline(train_df)
    if kind == "ollama":
        return Ollama(b["model"], b.get("url", "http://localhost:11434"), b.get("temperature", 0.0))
    if kind == "openai":
        return OpenAICompatible(b["model"], b["base_url"], b.get("api_key_env", "OPENAI_API_KEY"),
                                b.get("temperature", 0.0))
    if kind == "mlx":
        return MLX(b["model"], b.get("adapter"))
    raise ValueError(f"unknown backend {kind}")
