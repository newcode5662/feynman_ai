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
        """鲁棒的 JSON 解析"""
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

        return {
            "overall_score": 0.5,
            "dimensions": {"accuracy": 5, "clarity": 5, "completeness": 5, "examples": 5},
            "key_points": {"total": 3, "matched": 1, "list": ["解析失败"]},
            "feedback": "AI 评分解析出现问题，已给予中等分数",
            "teacher_comment": "请重新尝试",
            "ref_answer": "暂无参考答案",
            "missing_points": []
        }

    def _get_mastery_level(self, score: float) -> dict:
        """根据分数返回掌握等级"""
        for level_id, level_info in MASTERY_LEVELS.items():
            if score >= level_info['min']:
                return {
                    "level_id": level_id,
                    "label": level_info['label'],
                    "color": level_info['color'],
                    "desc": level_info['desc']
                }
        return MASTERY_LEVELS['beginner']

    def extract_key_points(self, content: str) -> list:
        """
        【新增】从知识内容中提取关键点
        """
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""请从以下知识内容中提取3-5个核心关键点。

【知识内容】：
{content}

请按以下JSON格式输出（只输出JSON）：
{{
    "key_points": [
        {{"point": "关键点1", "importance": "核心概念"}},
        {{"point": "关键点2", "importance": "重要细节"}},
        {{"point": "关键点3", "importance": "应用场景"}}
    ]
}}"""
        )

        try:
            res = self.llm.invoke(prompt.format(content=content))
            parsed = self._parse_llm_json(res)
            return parsed.get('key_points', [])
        except:
            return [{"point": "知识点提取失败", "importance": "未知"}]

    def generate_question(self, knowledge: dict) -> str:
        """生成引导性问题"""
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""你是一位费曼学习法教练。请阅读以下知识片段：

---
{content}
---

请生成一个引导性问题，帮助学生用自己的话解释这个概念。

要求：
1. 问题要像一个好奇的初学者问出来的
2. 不要直接暴露定义，而是引导学生思考和重构
3. 可以问"为什么"、"怎么理解"、"能举个例子吗"
4. 直接输出问题，不要有多余的话"""
        )
        return self.llm.invoke(prompt.format(content=knowledge["content"])).strip()

    def evaluate_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        """
        【重构】全面评估学生解释
        返回：分数 + 等级 + 关键点对照 + 教师评语 + 参考答案
        """
        prompt = PromptTemplate(
            input_variables=["original", "explanation"],
            template="""你是一位经验丰富的费曼学习法导师。请仔细对比原始知识和学生的解释。

【原始知识】：
{original}

【学生解释】：
{explanation}

请进行深度评估，严格按以下JSON格式输出（只输出JSON）：
{{
    "dimensions": {{
        "accuracy": 0-10,
        "clarity": 0-10,
        "completeness": 0-10,
        "examples": 0-10
    }},
    "overall_score": 0.0-1.0,
    "key_points": {{
        "total": 3,
        "matched": 2,
        "list": [
            {{"point": "关键点1", "matched": true, "student_said": "学生对应表述"}},
            {{"point": "关键点2", "matched": false, "student_said": "未提及"}}
        ]
    }},
    "teacher_comment": "像老师一样的点评，50-100字，温和但有建设性",
    "strengths": ["优点1", "优点2"],
    "missing_points": ["遗漏点1", "遗漏点2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "ref_answer": "一个完美的费曼式解释，使用简单类比，像给初学者讲课一样，100-150字"
}}

评分标准：
- accuracy（准确性）：核心概念是否正确
- clarity（清晰度）：是否通俗易懂，有无使用类比
- completeness（完整性）：关键点覆盖程度
- examples（举例能力）：是否能举出恰当例子"""
        )

        res = self.llm.invoke(
            prompt.format(
                original=knowledge["content"],
                explanation=user_explanation,
            )
        )

        eval_result = self._parse_llm_json(res)

        # 添加掌握等级
        score = eval_result.get('overall_score', 0.5)
        eval_result['mastery_level'] = self._get_mastery_level(score)

        # 计算各维度百分制
        dims = eval_result.get('dimensions', {})
        eval_result['dimensions_pct'] = {
            k: v * 10 for k, v in dims.items()
        }

        return eval_result

    def study_session(self, subject: str = None, mode: str = "sequential", specific_id: str = None) -> dict:
        """
        【重构】支持多种学习模式的会话
        mode: sequential / review / random / weak / specific
        """
        knowledge = None
        session_mode = mode
        position_info = ""

        # 1. 指定ID模式
        if specific_id:
            knowledge = self.kb.get_knowledge_by_id(specific_id)
            if not knowledge:
                return {"error": "未找到指定知识点"}
            session_mode = "精修"
            position_info = f"来源: {knowledge['metadata'].get('source', '未知')}"

        # 2. 复习模式（优先）
        elif mode == "review":
            due = self.tracker.get_due_reviews(limit=1)
            if due:
                kid, subj, preview, mastery = due[0]
                knowledge = self.kb.get_knowledge_by_id(kid)
                if knowledge:
                    session_mode = "复习"
                    position_info = f"上次掌握度: {int(mastery*100)}%"

        # 3. 顺序学习模式
        elif mode == "sequential":
            knowledge = self.kb.get_next_unlearned(subject, self.tracker)
            if knowledge:
                session_mode = "新学"
                position_info = knowledge.get('position', '')
            else:
                # 已学完，转为复习
                due = self.tracker.get_due_reviews(limit=1)
                if due:
                    kid, subj, preview, mastery = due[0]
                    knowledge = self.kb.get_knowledge_by_id(kid)
                    session_mode = "复习"
                    position_info = "已学完全部内容，进入复习"

        # 4. 薄弱点模式
        elif mode == "weak":
            weak_list = self.kb.get_weak_points(subject, self.tracker, limit=1)
            if weak_list:
                knowledge = weak_list[0]
                session_mode = "薄弱强化"
                position_info = f"上次得分: {int(knowledge.get('last_score', 0)*100)}%"

        # 5. 随机模式
        elif mode == "random":
            knowledge = self.kb.get_random_knowledge(subject)
            if knowledge:
                session_mode = "随机"

        # 最终检查
        if not knowledge:
            return {"error": "暂无可学习的知识，请先导入文档或切换学习模式"}

        # 生成问题
        question = self.generate_question(knowledge)

        # 提取关键点（供后续对照）
        key_points = self.extract_key_points(knowledge['content'])

        return {
            "mode": session_mode,
            "knowledge": knowledge,
            "question": question,
            "key_points": key_points,
            "position_info": position_info,
            "can_view_content": True  # 允许查看原文
        }

    def submit_explanation(self, knowledge: dict, user_explanation: str) -> dict:
        """提交解释并获取评估结果"""
        eval_res = self.evaluate_explanation(knowledge, user_explanation)

        k_id = knowledge.get("id") or str(hash(knowledge["content"]))

        # 记录学习数据
        self.tracker.record_review(
            k_id,
            knowledge.get("metadata", {}).get("subject", "默认"),
            knowledge["content"],
            eval_res.get("overall_score", 0),
            user_explanation,
            eval_res.get("teacher_comment", ""),
        )

        return eval_res
