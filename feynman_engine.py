import json
import re
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from config import LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_CONTEXT_WINDOW
from knowledge_base import KnowledgeBase
from progress_tracker import ProgressTracker

class FeynmanEngine:
    def __init__(self):
        os.environ['OLLAMA_BASE_URL'] = OLLAMA_BASE_URL
        # LLM 对象通常支持 num_ctx 参数，直接设置以扩大上下文
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.4,
            num_ctx=OLLAMA_CONTEXT_WINDOW
        )
        self.kb = KnowledgeBase()
        self.tracker = ProgressTracker()

    def _parse_llm_json(self, text: str) -> dict:
        """鲁棒的 JSON 解析"""
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
        return {}

    def analyze_file_content(self, preview_text: str) -> dict:
        """导入文件时分析领域和摘要"""
        # 【防御性截断】确保不发生 Decode Error
        safe_input = preview_text[:2500]

        prompt = PromptTemplate(
            input_variables=["content"],
            template="""你是一位博学的图书管理员。请阅读以下文件片段：
---
{content}
---
请提取以下信息并以JSON格式返回：
1. "domain": 该内容属于哪个具体专业领域？（如：量子物理、中国历史、Python编程）
2. "summary": 用一段话概括这份文档的核心教学内容（150字以内）。

输出JSON示例：
{{ "domain": "领域名", "summary": "摘要内容" }}
"""
        )
        try:
            res = self.llm.invoke(prompt.format(content=safe_input))
            return self._parse_llm_json(res)
        except Exception as e:
            print(f"Summary Error: {e}")
            return {"domain": "通用知识", "summary": "自动摘要生成失败"}

    def generate_question(self, knowledge: dict, domain: str = "通用") -> dict:
        prompt = PromptTemplate(
            input_variables=["domain", "content"],
            template="""你是一位【{domain}】领域的资深专家导师。
请阅读以下知识片段：
---
{content}
---
任务：
1. 提取这段内容的一个核心“知识点标签”（Topic Tag）。
2. 基于该知识点，向学生提出一个**概念性问题**。
   - 必须用中文提问。
   - 不要问“是什么”，要问“为什么”或“底层逻辑”。

请严格按JSON返回：
{{
    "topic_tag": "核心知识点",
    "question": "你的专家级提问"
}}
"""
        )
        try:
            res = self.llm.invoke(prompt.format(domain=domain, content=knowledge['content']))
            parsed = self._parse_llm_json(res)
            return {
                "question": parsed.get("question", "请解释这段内容的核心概念。"),
                "topic_tag": parsed.get("topic_tag", "知识点练习")
            }
        except:
            return {
                "question": f"请解释：{knowledge['content'][:50]}...",
                "topic_tag": "随机练习"
            }

    def evaluate_explanation(self, knowledge: dict, user_explanation: str, domain: str) -> dict:
        prompt = PromptTemplate(
            input_variables=["domain", "original", "explanation"],
            template="""你是一位【{domain}】领域的费曼学习法导师。

【原始教材】：
{original}

【学生解释】：
{explanation}

请执行以下评估：
1. **评分**：准确性、简单性、完整性。
2. **费曼式参考解释**：
   - 必须**基于【原始教材】**。
   - 必须使用**生活中的类比**或**极简的语言**。

请严格输出JSON：
{{
    "overall_score": 0.0-1.0,
    "feedback": "简短点评（中文）",
    "feynman_explanation": "你的类比式参考解释"
}}"""
        )
        try:
            res = self.llm.invoke(prompt.format(
                domain=domain, original=knowledge['content'], explanation=user_explanation
            ))
            return self._parse_llm_json(res)
        except:
            return {"overall_score": 0.0, "feedback": "服务繁忙，无法评分", "feynman_explanation": "无"}

    def study_session(self, subject: str = None, specific_id: str = None) -> dict:
        if specific_id:
            knowledge = self.kb.get_knowledge_by_id(specific_id)
            mode = "定向精修"
        else:
            due = self.tracker.get_due_reviews(limit=1)
            if due and (not subject or subject == "全部"):
                kid, subj, _, _ = due[0]
                knowledge = self.kb.get_knowledge_by_id(kid)
                mode = "复习模式"
            else:
                knowledge = self.kb.get_random_knowledge(subject)
                mode = "探索模式"

        if not knowledge: return {"error": "暂无知识，请先导入文档。"}

        # 获取元数据
        source = knowledge.get('metadata', {}).get('source', '')
        meta = self.tracker.get_file_metadata(source)
        domain = meta.get('domain', '通用知识')

        gen_res = self.generate_question(knowledge, domain)
        return {
            "mode": mode, "knowledge": knowledge, "domain": domain,
            "topic_tag": gen_res['topic_tag'], "question": gen_res['question']
        }

    def submit_explanation(self, knowledge: dict, user_explanation: str, domain: str) -> dict:
        eval_res = self.evaluate_explanation(knowledge, user_explanation, domain)
        k_id = knowledge.get('id')
        self.tracker.record_review(
            k_id, knowledge.get('metadata', {}).get('subject', '默认'),
            knowledge['content'], eval_res.get('overall_score', 0),
            user_explanation, eval_res.get('feedback', '')
        )
        return eval_res
