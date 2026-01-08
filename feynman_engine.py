import json
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from config import LLM_MODEL, OLLAMA_BASE_URL
from knowledge_base import KnowledgeBase
from progress_tracker import ProgressTracker

class FeynmanEngine:
    def __init__(self):
        self.llm = OllamaLLM(
            model=LLM_MODEL, 
            temperature=0.7,
            base_url=OLLAMA_BASE_URL
        )
        self.kb = KnowledgeBase()
        self.tracker = ProgressTracker()
    
    def _parse_llm_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                try:
                    return json.loads(json_str)
                except:
                    pass
        return {
            "accuracy": 0.0, "simplicity": 0.0, "completeness": 0.0, "examples": 0.0,
            "overall_score": 0.0,
            "feedback": f"解析失败，Raw: {text[:50]}...",
            "correct_points": [], "improve_points": ["格式错误"], "simple_explanation": ""
        }

    def generate_question(self, knowledge: dict) -> str:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""你是一位费曼学习法教练。根据以下知识片段，生成一个引导性问题。
            
知识内容：
{content}

要求：开放性问题，引导学生用简单的语言解释。直接输出问题。"""
        )
        return self.llm.invoke(prompt.format(content=knowledge['content'])).strip()
    
    def evaluate_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        prompt = PromptTemplate(
            input_variables=["original", "explanation"],
            template="""评估学生的解释。
            
原始知识：
{original}

学生解释：
{explanation}

请严格按以下 JSON 格式输出（只输出JSON）：
{{
    "accuracy": 0.0-1.0,
    "simplicity": 0.0-1.0, 
    "completeness": 0.0-1.0,
    "examples": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "feedback": "简短建议",
    "correct_points": ["点1"],
    "improve_points": ["点1"],
    "simple_explanation": "参考解释"
}}"""
        )
        res = self.llm.invoke(prompt.format(original=knowledge['content'], explanation=user_explanation))
        return self._parse_llm_json(res)
    
    def study_session(self, subject: str = None) -> dict:
        due = self.tracker.get_due_reviews(limit=1)
        if due:
            kid, subj, preview, _ = due[0]
            knowledge = {'id': kid, 'content': preview + "...", 'metadata': {'subject': subj}}
            mode = "复习"
        else:
            knowledge = self.kb.get_random_knowledge(subject)
            if not knowledge: return {"error": "暂无知识，请先导入文档"}
            mode = "新学"
        question = self.generate_question(knowledge)
        return {"mode": mode, "knowledge": knowledge, "question": question}

    def submit_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        eval_res = self.evaluate_explanation(knowledge, user_explanation)
        k_id = knowledge.get('id') or str(hash(knowledge['content']))
        self.tracker.record_review(
            k_id, knowledge.get('metadata', {}).get('subject', '默认'),
            knowledge['content'], eval_res['overall_score'],
            user_explanation, eval_res['feedback']
        )
        return eval_res
