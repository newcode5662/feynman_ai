import streamlit as st
import pandas as pd
import os
from document_processor import DocumentProcessor
from knowledge_base import KnowledgeBase
from feynman_engine import FeynmanEngine
from progress_tracker import ProgressTracker
from config import DOCUMENTS_DIR, MASTERY_LEVELS, LEARNING_MODES

st.set_page_config(page_title="费曼 AI 导师", page_icon="🎓", layout="wide")

# CSS 样式
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .mastery-card {
        padding: 15px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px;
    }
    .key-point-pass { color: #4CAF50; font-weight: bold; }
    .key-point-fail { color: #FF5252; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_core():
    return {
        'kb': KnowledgeBase(),
        'engine': FeynmanEngine(),
        'tracker': ProgressTracker()
    }

try:
    core = get_core()
except Exception as e:
    st.error(f"核心组件初始化失败: {e}")
    st.stop()

# 侧边栏
with st.sidebar:
    st.title("🎓 费曼 AI 导师")
    all_subjects = core['kb'].get_all_subjects()
    current_subject = st.selectbox("📚 当前课程", all_subjects) if all_subjects else "默认"
    st.divider()
    page = st.radio("导航", ["🗺️ 课程地图", "✍️ 开始学习", "📊 数据看板", "📂 资料导入"])

# 1. 课程地图
if page == "🗺️ 课程地图":
    st.header(f"🗺️ 学习路径：{current_subject}")
    course_data = core['kb'].get_chapter_progress(current_subject, core['tracker'])

    if not course_data['chapters']:
        st.info("👈 该课程暂无内容，请去「资料导入」页面上传文档")

    for chapter in course_data['chapters']:
        with st.expander(f"📖 {chapter['title']} ({chapter['stats']['completed']}/{chapter['stats']['total']})", expanded=True):
            st.progress(chapter['stats']['progress_pct'] / 100)
            for chunk in chapter['chunks']:
                c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
                score = chunk['progress']['last_score'] if chunk['progress'] else 0
                status = "🟢" if score >= 0.9 else "🟡" if score >= 0.6 else "🔴" if chunk['progress'] else "⚪"

                with c1: st.text(status)
                with c2: st.caption(chunk['preview'])
                with c3:
                    if st.button("学习", key=f"btn_{chunk['id']}"):
                        st.session_state.target_id = chunk['id']
                        st.switch_page("app.py") # 注意：本地运行可能需要改为 st.rerun() 或提示用户切换Tab
                        # 如果 switch_page 报错，可以用 st.info("请切换到「✍️ 开始学习」页面，已自动选中")

# 2. 开始学习
elif page == "✍️ 开始学习":
    st.header("✍️ 费曼深度学习")
    c1, c2 = st.columns([3, 1])
    with c1:
        mode_key = st.selectbox("选择模式", list(LEARNING_MODES.keys()), format_func=lambda x: LEARNING_MODES[x])
    with c2:
        if st.button("🚀 下一题", type="primary"):
            st.session_state.study_data = None
            st.session_state.eval_result = None
            st.rerun()

    if 'study_data' not in st.session_state: st.session_state.study_data = None
    if 'eval_result' not in st.session_state: st.session_state.eval_result = None

    # 获取题目
    if not st.session_state.study_data:
        target_id = st.session_state.get('target_id')
        if target_id:
            res = core['engine'].study_session(current_subject, mode="specific", specific_id=target_id)
            st.session_state.target_id = None
        else:
            res = core['engine'].study_session(current_subject, mode=mode_key)

        if "error" in res:
            st.error(res['error'])
        else:
            st.session_state.study_data = res
            st.rerun()

    if st.session_state.study_data:
        data = st.session_state.study_data
        st.caption(f"模式: {data.get('mode')} | {data.get('position_info', '')}")
        st.markdown(f"### Q: {data['question']}")

        with st.expander("🔍 查看原文线索"):
            st.info(data['knowledge']['content'])

        user_input = st.text_area("你的解释:", height=150)

        if st.button("提交评估") and user_input:
            with st.spinner("AI 正在批改..."):
                res = core['engine'].submit_explanation(data['knowledge'], user_input)
                st.session_state.eval_result = res

        if st.session_state.eval_result:
            r = st.session_state.eval_result
            st.divider()
            lvl = r.get('mastery_level', MASTERY_LEVELS['beginner'])
            st.markdown(f"""
            <div class="mastery-card" style="background-color: {lvl['color']};">
                <h1>{int(r['overall_score']*100)}分 - {lvl['label']}</h1>
                <p>{r.get('teacher_comment', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            # 关键点对照
            st.subheader("🎯 关键点检查")
            kp = r.get('key_points', {})
            if kp.get('list'):
                for point in kp['list']:
                    icon = "✅" if point.get('matched') else "❌"
                    color = "key-point-pass" if point.get('matched') else "key-point-fail"
                    st.markdown(f"- {icon} <span class='{color}'>{point['point']}</span>", unsafe_allow_html=True)

            with st.expander("📚 参考答案"):
                st.write(r.get('ref_answer'))

# 3. 数据看板
elif page == "📊 数据看板":
    stats = core['tracker'].get_statistics()
    k1, k2 = st.columns(2)
    k1.metric("总知识点", stats['total_knowledge'])
    k2.metric("平均掌握度", f"{stats['avg_mastery']}%")

# 4. 资料导入
elif page == "📂 资料导入":
    st.header("📚 导入文档")
    f = st.file_uploader("支持 PDF/Docx/MD", type=['pdf', 'docx', 'md'])
    sub = st.text_input("课程名称", value="未命名课程")
    if f and st.button("导入"):
        path = os.path.join(DOCUMENTS_DIR, f.name)
        with open(path, "wb") as file: file.write(f.getbuffer())
        with st.spinner("正在后台分批处理..."):
            count = core['kb'].add_document(path, sub)
            st.success(f"成功导入 {count} 个知识块！")
