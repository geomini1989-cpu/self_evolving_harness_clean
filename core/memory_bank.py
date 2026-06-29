import json
import os
import random
import re


class MemoryBank:
    CATEGORY_KEYWORDS = {
        "\u9000\u6b3e\u7ea0\u7eb7": [
            "\u9000\u6b3e", "\u9000\u94b1", "\u9000\u8d27", "\u8d54", "\u8d54\u507f",
            "\u8865\u507f", "\u552e\u540e", "\u574f\u4e86", "\u4e0d\u80fd\u7528",
            "\u8d28\u91cf", "\u7834\u635f", "\u53d1\u9709", "\u53d7\u6f6e",
            "\u96be\u5403", "\u51c9\u4e86", "\u4e0d\u65b0\u9c9c", "\u5c11\u9001",
        ],
        "\u7269\u6d41\u6295\u8bc9": [
            "\u5feb\u9012", "\u7269\u6d41", "\u9a91\u624b", "\u9a91\u58eb", "\u5916\u5356",
            "\u914d\u9001", "\u9001\u9910", "\u9001\u9519", "\u8fdf\u5230", "\u8d85\u65f6",
            "\u665a\u4e86", "\u6162", "\u53f8\u673a", "\u6001\u5ea6", "\u4e22\u4ef6",
        ],
        "\u8d26\u53f7\u5c01\u7981": [
            "\u8d26\u53f7", "\u8d26\u6237", "\u5c01\u7981", "\u5c01\u53f7",
            "\u89e3\u5c01", "\u767b\u5f55", "\u767b\u9646", "\u5bc6\u7801",
            "\u51bb\u7ed3", "\u5f02\u5730", "\u98ce\u63a7",
        ],
        "\u7cfb\u7edfBug": [
            "\u7cfb\u7edf", "bug", "\u5d29\u6e83", "\u95ea\u9000", "\u62a5\u9519",
            "\u6253\u4e0d\u5f00", "\u5361\u4f4f", "\u9875\u9762", "\u7f51\u7edc",
            "\u9a8c\u8bc1\u7801", "\u652f\u4ed8\u5931\u8d25", "\u5ba2\u670d\u6ca1\u7528",
        ],
        "\u865a\u5047\u5ba3\u4f20": [
            "\u865a\u5047", "\u5ba3\u4f20", "\u5e7f\u544a", "\u56fe\u6587\u4e0d\u7b26",
            "\u4e0d\u7b26", "\u5047\u8d27", "\u6b3a\u9a97", "\u5938\u5927",
            "\u627f\u8bfa", "\u6d3b\u52a8", "\u4e0d\u503c\u5f97\u4fe1\u4efb",
        ],
    }
    DEFAULT_MESSAGE = "No matched category-specific skills. Use the base schema and the input text."
    EMPTY_MESSAGE = "No business skills are currently available. Use the base schema and the input text."

    def __init__(self):
        self.examples_file = "memory/examples.json"
        self.examples = []
        self.skills_db = {}
        self._load_few_shots()
        self._parse_skills_from_md()

    def _load_few_shots(self):
        if os.path.exists(self.examples_file):
            try:
                with open(self.examples_file, "r", encoding="utf-8") as f:
                    self.examples = json.load(f)
            except Exception:
                self.examples = []
        else:
            self.examples = []

    def add_successful_case(self, input_text, output_json):
        if any(ex.get("input") == input_text for ex in self.examples):
            return
        self.examples.append({"input": input_text, "output": output_json})
        os.makedirs(os.path.dirname(self.examples_file), exist_ok=True)
        with open(self.examples_file, "w", encoding="utf-8") as f:
            json.dump(self.examples[-100:], f, ensure_ascii=False, indent=2)

    def _parse_skills_from_md(self):
        skill_file = "memory/SKILL.md"
        if not os.path.exists(skill_file):
            return
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(r"##\s*\[(.*?)\]\s*(.*?)\n(.*?)(?=\n##|\Z)", re.DOTALL)
        for category, title, body in pattern.findall(content):
            category = category.strip()
            title = title.strip()
            body = body.strip()
            self.skills_db.setdefault(category, []).append(f"[{title}]\n{body}")

    def refresh_skills(self):
        self.skills_db = {}
        self._parse_skills_from_md()

    def refresh_memory(self):
        self._load_few_shots()
        self.refresh_skills()

    def route_categories(self, texts, max_categories=3):
        scores = self.score_categories(texts)
        return [category for category, _ in scores[:max_categories]]

    def score_categories(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        joined_text = "\n".join(str(text) for text in texts).lower()
        scores = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword.lower() in joined_text)
            if score:
                scores.append((score, category))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [(category, score) for score, category in scores]

    def get_skills_by_categories(self, categories, top_k_per_cat=2, max_chars=1200):
        if not self.skills_db:
            return self.EMPTY_MESSAGE
        retrieved_skills = []
        seen = set()
        for category in categories:
            if category not in self.skills_db:
                continue
            for skill in self.skills_db[category][-top_k_per_cat:]:
                if skill in seen:
                    continue
                seen.add(skill)
                retrieved_skills.append(skill)
        if not retrieved_skills:
            return self.DEFAULT_MESSAGE
        text = "\n\n".join(retrieved_skills)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[truncated]"
        return text

    def get_few_shots(self, current_input, k=2, enabled=True, max_chars=1000):
        if not enabled or not self.examples:
            return ""
        sample_size = min(len(self.examples), k)
        chosen_examples = random.sample(self.examples, sample_size)
        lines = ["Reference successful cases:"]
        for i, ex in enumerate(chosen_examples, 1):
            output = json.dumps(ex.get("output", {}), ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{i}. input={ex.get('input', '')}\noutput={output}")
        prompt = "\n".join(lines)
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars].rstrip() + "\n[truncated]"
        return prompt
