import json
import os
import random
import re


class MemoryBank:
    CATEGORY_KEYWORDS = {
        '退款纠纷': ['退款', '退钱', '退货', '赔', '赔偿', '售后', '坏了', '不能用', '质量', '破损', '发霉', '受潮'],
        '物流投诉': ['快递', '物流', '骑手', '外卖', '配送', '送错', '迟到', '超时', '司机', '态度', '丢件', '派送'],
        '账号封禁': ['账号', '账户', '封禁', '封号', '解封', '登录', '登陆', '密码', '冻结', '异地', '风控'],
        '系统Bug': ['系统', 'bug', '崩溃', '闪退', '报错', '打不开', '卡住', '页面', '网络', '验证码', '支付失败'],
        '虚假宣传': ['虚假', '宣传', '广告', '图文不符', '不符', '假货', '欺骗', '夸大', '承诺', '活动'],
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

    def route_categories(self, texts, max_categories=3):
        if isinstance(texts, str):
            texts = [texts]
        joined_text = "\n".join(str(text) for text in texts).lower()
        scores = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword.lower() in joined_text)
            if score:
                scores.append((score, category))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [category for _, category in scores[:max_categories]]

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
