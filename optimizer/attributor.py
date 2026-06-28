import json
import re
import os

class SkillAttributor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        # 确保 memory 目录存在
        os.makedirs(os.path.dirname(self.skill_file), exist_ok=True)

    def analyze_root_cause(self, eval_result, current_input, prediction_text, ground_truth, max_retries=2):
        """修改：带有自我纠错重试机制的归因诊断"""
        base_prompt = f"""你是一个顶级的 AI 系统诊断引擎。以下是一次大模型在“复杂客诉解析”任务中的执行失败记录。
        【用户输入】：{current_input}
        【真实标签】：{json.dumps(ground_truth, ensure_ascii=False)}
        【模型输出】：{prediction_text}
        【评估报错】：{eval_result.get('error_reason')}

        请结构化诊断并返回严格的纯 JSON 对象，格式如下：
        {{
            "error_category": "填入错误类别",
            "root_cause_analysis": "简短的根本原因分析",
            "proposed_rule": "【强制规则】当遇到...情况时，必须..."
        }}"""
        
        current_prompt = base_prompt
        for attempt in range(max_retries):
            suggestion = self.llm.generate(current_prompt, model_type="smart")
            
            try:
                cleaned_suggestion = re.sub(r'\`{3}(?:json)?(.*?)\`{3}', r'\1', suggestion, flags=re.DOTALL).strip()
                if not cleaned_suggestion.startswith('{'):
                    match = re.search(r'\{.*?\}', cleaned_suggestion, re.DOTALL)
                    if match: cleaned_suggestion = match.group(0)
                        
                patch_data = json.loads(cleaned_suggestion)
                if "proposed_rule" in patch_data:
                    return patch_data
                raise ValueError("缺少 proposed_rule 字段")
            except Exception as e:
                print(f"⚠️ [Attributor] 解析 JSON 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 动态追加报错上下文进行重试
                    current_prompt = base_prompt + f"\n\n⚠️ 【警告】解析失败: {e}。请检查格式，务必只输出合法 JSON！\n你的错误输出：{suggestion}"
                else:
                    return {"error_category": "归因引擎解析失败", "root_cause_analysis": "未能返回合法 JSON", "proposed_rule": None}

    def apply_patch(self, patch_data):
        """
        微创级修补：不再全量覆盖，而是将新规则追加到技能库中
        """
        new_rule = patch_data.get("proposed_rule")
        if not new_rule:
            print("⏭️ [Attributor] 未生成有效的新规则，跳过本次修补。")
            return False
            
        print(f"📉 [病理诊断] 错误分类: {patch_data.get('error_category')}")
        print(f"💡 [根因分析] {patch_data.get('root_cause_analysis')}")
        print(f"🛠️ [微创修补] 准备打入新规则: {new_rule}")
        
        # 读取当前内容
        if os.path.exists(self.skill_file):
            with open(self.skill_file, "r", encoding="utf-8") as f:
                current_content = f.read().strip()
        else:
            current_content = "# 智能体核心执行规范 (SKILL.md)\n> 本文档由系统自进化闭环自动维护，请勿手动修改。\n\n## 业务规则列表："
            
        # 确保现有内容末尾有换行
        if not current_content.endswith('\n'):
            current_content += '\n'
            
        # 追加新规则（作为列表项）
        updated_content = current_content + f"- {new_rule}\n"
        
        # 写入文件 (注意这里用 'w' 是因为我们用 updated_content 拼接了原文，实现了实质上的 Append)
        with open(self.skill_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        print("✅ [Attributor] 规则补丁已成功追加至 SKILL.md，系统完成自我进化！")
        return True