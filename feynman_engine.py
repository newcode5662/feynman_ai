import json
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from config import LLM_MODEL, OLLAMA_BASE_URL, MASTERY_LEVELS
from knowledge_base import KnowledgeBase
from progress_tracker import ProgressTracker

class FeynmanEngine:
    def __init__(self):
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.3,
            base_url=OLLAMA_BASE_URL,
        )
        self.kb = KnowledgeBase()
        self.tracker = ProgressTracker()

    def _parse_llm_json(self, text: str) -> dict:
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        # 兜底返回，防止应用崩溃
        return {
            "overall_score": 0.5,
            "dimensions": {"accuracy": 5, "clarity": 5, "completeness": 5, "examples": 5},
            "key_points": {"list": []},
            "teacher_comment": "解析评分失败，但这不影响你的学习记录。",
            "ref_answer": "暂无标准答案"
        }

    def _get_mastery_level(self, score: float) -> dict:
        for level_id, level_info in MASTERY_LEVELS.items():
            if score >= level_info['min']:
                return level_id, level_info
        return "beginner", MASTERY_LEVELS['beginner']

    def extract_key_points(self, content: str) -> list:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""从以下内容提取3-5个核心关键词或短语。
内容：
{content}

请输出JSON格式：
{{ "key_points": [{{"point": "关键词1", "importance": "high"}}] }}
"""
        )
        try:
            res = self.llm.invoke(prompt.format(content=content))
            return self._parse_llm_json(res).get('key_points', [])
        except:
            return []

    def generate_question(self, knowledge: dict) -> str:
        # === 关键修复：增加语言自适应指令 ===
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""你是一位费曼学习法教练。请阅读以下知识片段：
---
{content}
---
请生成一个引导性问题。

⚠️ 重要规则：
1. **请始终用中文提问**
2. 问题要像初学者问的，不要直接暴露定义。
3. 直接输出问题本身
"""
        )
        try:
            return self.llm.invoke(prompt.format(content=knowledge["content"])).strip()
        except Exception as e:
            return f"请解释这段内容：{knowledge['content'][:50]}..."

    def evaluate_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        prompt = PromptTemplate(
            input_variables=["original", "explanation"],
            template="""你是一位严格的导师。请对比原始知识和学生的解释。

【原始知识】：
{original}

【学生解释】：
{explanation}

请严格按以下 JSON 格式输出（只输出JSON）：
{{
    "dimensions": {{
        "accuracy": 0-10,
        "clarity": 0-10,
        "completeness": 0-10,
        "examples": 0-10
    }},
    "overall_score": 0.0-1.0,
    "key_points": {{
        "list": [
            {{"point": "Key Concept 1", "matched": true, "student_said": "..."}},
            {{"point": "Key Concept 2", "matched": false, "student_said": "N/A"}}
        ]
    }},
    "teacher_comment": "简短点评（始终用中文点评）",
    "ref_answer": "参考解释（与原始内容语言一致）"
}}"""
        )

        try:
            res = self.llm.invoke(prompt.format(original=knowledge["content"], explanation=user_explanation))
            eval_result = self._parse_llm_json(res)
        except Exception as e:
            print(f"Eval Error: {e}")
            eval_result = self._parse_llm_json("{}")

        # 补全数据结构
        score = eval_result.get('overall_score', 0.5)
        _, level_info = self._get_mastery_level(score)
        eval_result['mastery_level'] = level_info

        dims = eval_result.get('dimensions', {})
        eval_result['dimensions_pct'] = {k: v * 10 for k, v in dims.items()}

        return eval_result

    def study_session(self, subject: str = None, mode: str = "sequential", specific_id: str = None) -> dict:
        knowledge = None
        session_mode = mode
        position_info = ""

        # 逻辑同前，增加 try-except 保护
        try:
            if specific_id:
                knowledge = self.kb.get_knowledge_by_id(specific_id)
                session_mode = "精修"
            elif mode == "review":
                due = self.tracker.get_due_reviews(limit=1)
                if due:
                    kid, subj, preview, mastery = due[0]
                    knowledge = self.kb.get_knowledge_by_id(kid)
                    session_mode = "复习"
            elif mode == "sequential":
                knowledge = self.kb.get_next_unlearned(subject, self.tracker)
                if knowledge:
                    session_mode = "新学"
                    position_info = knowledge.get('position', '')
                else:
                    knowledge = self.kb.get_random_knowledge(subject) # 兜底
                    session_mode = "随机"
            elif mode == "weak":
                weak = self.kb.get_weak_points(subject, self.tracker, 1)
                if weak: knowledge = weak[0]
            elif mode == "random":
                knowledge = self.kb.get_random_knowledge(subject)

            if not knowledge:
                return {"error": "没有找到可学习的知识点，请检查是否有导入文档。"}

            question = self.generate_question(knowledge)
            return {
                "mode": session_mode,
                "knowledge": knowledge,
                "question": question,
                "position_info": position_info,
                "can_view_content": True
            }
        except Exception as e:
            print(f"Session Error: {e}")
            return {"error": f"系统繁忙，请重试: {str(e)}"}

    def submit_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        eval_res = self.evaluate_explanation(knowledge, user_explanation)
        k_id = knowledge.get("id") or str(hash(knowledge["content"]))
        self.tracker.record_review(
            k_id,
            knowledge.get("metadata", {}).get("subject", "默认"),
            knowledge["content"],
            eval_res.get("overall_score", 0),
            user_explanation,
            eval_res.get("teacher_comment", ""),
        )
        return eval_res
