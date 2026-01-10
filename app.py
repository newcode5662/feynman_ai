import streamlit as st
import pandas as pd
import os
from document_processor import DocumentProcessor
from knowledge_base import KnowledgeBase
from feynman_engine import FeynmanEngine
from progress_tracker import ProgressTracker
from config import DOCUMENTS_DIR, MASTERY_LEVELS, LEARNING_MODES

# ========== 页面配置 ==========
st.set_page_config(
    page_title="费曼 AI 导师",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 样式注入 ==========
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .mastery-card {
        padding: 15px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px;
    }
    .big-font { font-size: 20px !important; }
    .key-point-pass { color: #4CAF50; font-weight: bold; }
    .key-point-fail { color: #FF5252; }
</style>
""", unsafe_allow_html=True)

# ========== 初始化核心组件 ==========
@st.cache_resource
def get_core():
    return {
        'kb': KnowledgeBase(),
        'engine': FeynmanEngine(),
        'tracker': ProgressTracker()
    }

core = get_core()

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("🎓 费曼 AI 导师")

    # 全局学科选择
    all_subjects = core['kb'].get_all_subjects()
    if not all_subjects:
        st.warning("暂无课程，请先导入")
        current_subject = "默认"
    else:
        current_subject = st.selectbox("📚 当前课程", all_subjects)

    st.divider()

    # 导航
    page = st.radio("导航", ["🗺️ 课程地图", "✍️ 开始学习", "📊 数据看板", "📂 资料导入"])

    st.divider()

    # 迷你状态
    stats = core['tracker'].get_statistics()
    st.caption(f"已掌握: {stats.get('mastered_count', 0)} / {stats.get('total_knowledge', 0)}")
    st.progress(stats.get('avg_mastery', 0) / 100)

# ========== 1. 课程地图 (Course Map) ==========
if page == "🗺️ 课程地图":
    st.header(f"🗺️ 学习路径：{current_subject}")

    # 获取章节结构与进度
    course_data = core['kb'].get_chapter_progress(current_subject, core['tracker'])

    if not course_data['chapters']:
        st.info("👈 该课程暂无内容，请去「资料导入」页面上传文档")

    for chapter in course_data['chapters']:
        with st.expander(f"📖 {chapter['title']} ({chapter['stats']['completed']}/{chapter['stats']['total']})", expanded=True):
            # 进度条
            st.progress(chapter['stats']['progress_pct'] / 100)

            # 知识点列表
            for chunk in chapter['chunks']:
                col1, col2, col3 = st.columns([0.1, 0.7, 0.2])

                # 状态图标
                status = "⚪"  # 未学
                score = 0
                if chunk['progress']:
                    score = chunk['progress']['last_score']
                    if score >= 0.9: status = "🟢"     # 精通
                    elif score >= 0.6: status = "🟡"   # 及格
                    else: status = "🔴"                # 需复习

                with col1: st.text(status)
                with col2: st.caption(chunk['preview'])
                with col3:
                    if st.button("进入学习", key=f"btn_{chunk['id']}"):
                        st.session_state.target_id = chunk['id']
                        st.session_state.target_subject = current_subject
                        st.switch_page("app.py") # 刷新重定向前需要处理逻辑，这里简单提示切换
                        st.info("请切换到「✍️ 开始学习」页面，已自动选中该知识点") # 实际逻辑需配合session state

# ========== 2. 开始学习 (Study Mode) ==========
elif page == "✍️ 开始学习":
    st.header("✍️ 费曼深度学习")

    # 学习模式选择
    c1, c2 = st.columns([3, 1])
    with c1:
        mode_key = st.selectbox("选择模式", list(LEARNING_MODES.keys()), format_func=lambda x: LEARNING_MODES[x])
    with c2:
        if st.button("🚀 开始新会话", type="primary", use_container_width=True):
            st.session_state.study_data = None
            st.session_state.eval_result = None
            st.rerun()

    # 初始化 Session State
    if 'study_data' not in st.session_state: st.session_state.study_data = None
    if 'eval_result' not in st.session_state: st.session_state.eval_result = None

    # 获取题目逻辑
    if not st.session_state.study_data:
        with st.spinner("AI 导师正在准备教案..."):
            # 检查是否有来自地图页的跳转
            target_id = st.session_state.get('target_id')
            if target_id:
                res = core['engine'].study_session(current_subject, mode="specific", specific_id=target_id)
                st.session_state.target_id = None # 清除跳转标记
            else:
                res = core['engine'].study_session(current_subject, mode=mode_key)

            if "error" in res:
                st.error(res['error'])
            else:
                st.session_state.study_data = res

    # 学习界面
    if st.session_state.study_data:
        data = st.session_state.study_data

        # 顶部信息栏
        st.caption(f"当前模式: {LEARNING_MODES.get(data.get('mode', 'random'), data.get('mode'))} | 📍 {data.get('position_info', '')}")

        # 问题区
        st.markdown(f"### Q: {data['question']}")

        # 【功能2需求：回答时可查看原文】
        with st.expander("🔍 遇到困难？点击查看原文线索"):
            st.info("提示：尝试先不看原文作答，效果更好哦！")
            st.markdown(f"**原文内容：**\n\n{data['knowledge']['content']}")

        # 作答区
        user_input = st.text_area("你的费曼解释 (试着像教别人一样说出来):", height=200, placeholder="例如：这就好比...")

        if st.button("提交评估", type="primary"):
            if not user_input:
                st.warning("请先输入你的解释")
            else:
                with st.spinner("👩‍🏫 老师正在认真批改..."):
                    res = core['engine'].submit_explanation(data['knowledge'], user_input)
                    st.session_state.eval_result = res

        # 结果展示区 (重构核心)
        if st.session_state.eval_result:
            r = st.session_state.eval_result
            st.divider()

            # 1. 评分与等级卡片
            lvl = r.get('mastery_level', MASTERY_LEVELS['beginner'])
            st.markdown(f"""
            <div class="mastery-card" style="background-color: {lvl['color']};">
                <h1>{int(r['overall_score']*100)}分</h1>
                <h2>{lvl['label']}</h2>
                <p>{lvl['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            # 2. 教师评语
            st.chat_message("assistant").write(f"**👩‍🏫 导师点评：** {r.get('teacher_comment', '暂无点评')}")

            # 3. 维度雷达 (用进度条模拟)
            st.subheader("📊 维度分析")
            dcols = st.columns(4)
            dims = r.get('dimensions_pct', {})
            labels = {'accuracy': '准确性', 'clarity': '清晰度', 'completeness': '完整度', 'examples': '举例'}
            for i, (k, v) in enumerate(labels.items()):
                with dcols[i]:
                    st.metric(v, f"{int(dims.get(k, 0))}/10")
                    st.progress(dims.get(k, 0)/100)

            # 4. 【功能3需求：关键点对照】
            st.subheader("🎯 关键点对照")
            kp = r.get('key_points', {})
            if kp.get('list'):
                for point in kp['list']:
                    icon = "✅" if point.get('matched') else "❌"
                    color_cls = "key-point-pass" if point.get('matched') else "key-point-fail"
                    st.markdown(f"- {icon} <span class='{color_cls}'>{point['point']}</span>", unsafe_allow_html=True)
                    if not point.get('matched'):
                        st.caption(f"   💡 建议补充: {point.get('student_said', '未提及')}")

            # 5. 参考答案
            with st.expander("📚 查看完美解释 (参考答案)"):
                st.success(r.get('ref_answer', '暂无参考答案'))

            if st.button("下一题 ➡️"):
                st.session_state.study_data = None
                st.session_state.eval_result = None
                st.rerun()

# ========== 3. 数据看板 ==========
elif page == "📊 数据看板":
    st.header("📈 学习数据中心")
    stats = core['tracker'].get_statistics()

    k1, k2, k3 = st.columns(3)
    k1.metric("总知识点", stats['total_knowledge'])
    k2.metric("已完全掌握", stats['mastered_count'])
    k3.metric("平均掌握度", f"{stats['avg_mastery']}%")

    st.subheader("学科概览")
    if stats['by_subject']:
        df = pd.DataFrame(stats['by_subject'])
        st.dataframe(
            df.style.highlight_max(axis=0, subset=['mastery']),
            column_config={
                "subject": "学科",
                "total": "总条目",
                "mastery": st.column_config.ProgressColumn("平均掌握度", format="%.1f%%", min_value=0, max_value=100),
                "mastered": "精通数量"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("暂无数据")

    st.subheader("近期活跃")
    if stats['weekly_stats']:
        wdf = pd.DataFrame(stats['weekly_stats'], columns=['日期', '总复习', '有效复习'])
        st.line_chart(wdf.set_index('日期'))

# ========== 4. 资料导入 (保持原有逻辑) ==========
elif page == "📂 资料导入":
    st.header("📚 知识库管理")
    uploaded_file = st.file_uploader("上传文档 (PDF/Word/MD)", type=['pdf', 'docx', 'md', 'txt'])
    subject_inp = st.text_input("归属学科/课程名称", value="未命名课程")

    if uploaded_file and st.button("开始处理", type="primary"):
        save_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("正在构建课程结构...", expanded=True) as status:
            st.write("📂 解析文档中...")
            count = core['kb'].add_document(save_path, subject_inp)
            st.write(f"🧠 向量化 {count} 个知识块完成...")
            status.update(label="✅ 导入成功！", state="complete", expanded=False)

        st.success(f"成功导入《{uploaded_file.name}》到【{subject_inp}】课程中！")
        st.balloons()
