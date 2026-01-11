import streamlit as st
import os
import pandas as pd
from knowledge_base import KnowledgeBase
from feynman_engine import FeynmanEngine
from progress_tracker import ProgressTracker
from config import DOCUMENTS_DIR

st.set_page_config(page_title="费曼 AI 专家版", page_icon="🎓", layout="wide")
st.markdown("""<style>.stButton>button {width: 100%;} .info-card {padding:15px; background-color:#f0f2f6; border-radius:10px; margin-bottom:15px;} .tag {background-color:#e0e0e0; padding:2px 8px; border-radius:4px; font-size:0.8em;}</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_core():
    # 延迟加载，防止启动时报错
    return {'kb': KnowledgeBase(), 'engine': FeynmanEngine(), 'tracker': ProgressTracker()}

try:
    core = get_core()
except Exception as e:
    st.error(f"⚠️ 系统启动失败: {e}")
    st.stop()

# 状态初始化
if 'page' not in st.session_state: st.session_state.page = "dashboard"
if 'current_subject' not in st.session_state: st.session_state.current_subject = "全部"
if 'study_session' not in st.session_state: st.session_state.study_session = None
if 'eval_result' not in st.session_state: st.session_state.eval_result = None
if 'target_id' not in st.session_state: st.session_state.target_id = None

# SPA 路由函数
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

with st.sidebar:
    st.title("🎓 费曼 AI")
    if st.button("📊 学习概览", type="primary" if st.session_state.page=="dashboard" else "secondary"): navigate_to("dashboard")
    if st.button("🗺️ 知识地图", type="primary" if st.session_state.page=="map" else "secondary"): navigate_to("map")
    if st.button("✍️ 开始学习", type="primary" if st.session_state.page=="study" else "secondary"):
        if st.session_state.page != "study": navigate_to("study")
    if st.button("📂 导入资料", type="primary" if st.session_state.page=="import" else "secondary"): navigate_to("import")
    st.divider()
    st.session_state.current_subject = st.selectbox("📚 专注学科", ["全部"] + core['kb'].get_all_subjects())

# ========== 页面：导入资料 (带生成器进度条) ==========
if st.session_state.page == "import":
    st.header("📂 导入与分析")
    uploaded_file = st.file_uploader("支持 PDF/Word/MD", type=['pdf', 'docx', 'md', 'txt'])
    subject_input = st.text_input("学科分类", value="通用")

    if uploaded_file and st.button("🚀 智能导入"):
        save_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())

        # 进度条容器
        progress_bar = st.progress(0, text="初始化中...")

        try:
            # 调用生成器
            generator = core['kb'].add_document(save_path, subject_input)

            total_count = 0
            preview_text = ""

            # 消费生成器
            for result in generator:
                if len(result) == 3: # 进度更新
                    prog, total, msg = result
                    progress_bar.progress(prog, text=msg)
                else: # 最终结果
                    total_count, preview_text = result

            progress_bar.progress(1.0, text="✅ 向量化完成！正在进行 AI 分析...")

            # AI 分析
            analysis = core['engine'].analyze_file_content(preview_text)
            core['tracker'].save_file_metadata(uploaded_file.name, analysis.get('domain'), analysis.get('summary'))

            st.success(f"成功导入 {total_count} 个知识块！")
            st.markdown(f"<div class='info-card'><b>识别领域：</b>{analysis.get('domain')}<br><b>摘要：</b>{analysis.get('summary')}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"导入出错: {e}")

# ========== 页面：知识地图 ==========
elif st.session_state.page == "map":
    st.header(f"🗺️ 知识地图 ({st.session_state.current_subject})")
    subj = st.session_state.current_subject if st.session_state.current_subject != "全部" else None
    if not subj: st.warning("请选择具体学科")
    else:
        structure = core['kb'].get_course_structure(subj)
        if not structure: st.info("暂无数据")
        for filename, chunks in structure.items():
            meta = core['tracker'].get_file_metadata(filename)
            with st.expander(f"📄 {filename} - [{meta['domain']}]"):
                st.caption(meta['summary'])
                for chunk in chunks:
                    c1, c2 = st.columns([0.8, 0.2])
                    c1.text(f"片段 {chunk['chunk_id']}: {chunk['preview']}")
                    if c2.button("去学习", key=f"btn_{chunk['id']}"):
                        st.session_state.target_id = chunk['id']
                        st.session_state.study_session = None
                        st.session_state.eval_result = None
                        navigate_to("study")

# ========== 页面：学习模式 ==========
elif st.session_state.page == "study":
    st.header("✍️ 费曼深度学习")
    if st.session_state.study_session is None:
        target = st.session_state.target_id
        subj = st.session_state.current_subject
        with st.spinner("🧠 专家出题中..."):
            res = core['engine'].study_session(subject=subj, specific_id=target)
            if "error" in res: st.error(res['error'])
            else:
                st.session_state.study_session = res
                st.session_state.target_id = None
                st.rerun()

    if st.session_state.study_session:
        data = st.session_state.study_session
        st.markdown(f"<div class='info-card'><span class='tag'>模式: {data['mode']}</span> <span class='tag'>领域: {data['domain']}</span> <span class='tag' style='background:#d4edda;color:#155724'>🎯 {data['topic_tag']}</span></div>", unsafe_allow_html=True)
        st.subheader(f"Q: {data['question']}")
        with st.expander("🔍 查看原文"): st.info(data['knowledge']['content'])
        user_input = st.text_area("你的解释:", height=200)
        c1, c2 = st.columns([1, 1])
        if c1.button("提交评估", type="primary"):
            if not user_input: st.warning("请先输入")
            else:
                with st.spinner("批改中..."):
                    res = core['engine'].submit_explanation(data['knowledge'], user_input, data['domain'])
                    st.session_state.eval_result = res
        if c2.button("下一题"):
            st.session_state.study_session = None
            st.session_state.eval_result = None
            st.rerun()

        if st.session_state.eval_result:
            r = st.session_state.eval_result
            st.divider()
            score = r.get('overall_score', 0)
            color = "#28a745" if score >= 0.8 else "#dc3545"
            st.markdown(f"<h3 style='color:{color}'>得分: {int(score*100)}</h3>", unsafe_allow_html=True)
            st.info(f"👨‍🏫 点评: {r.get('feedback')}")
            st.success(f"💡 参考: {r.get('feynman_explanation')}")

# ========== 页面：数据看板 ==========
elif st.session_state.page == "dashboard":
    st.header("📊 数据看板")
    stats = core['tracker'].get_statistics()
    k1, k2, k3 = st.columns(3)
    k1.metric("知识总量", stats['total_knowledge'])
    k2.metric("已精通", stats.get('mastered_count', 0))
    k3.metric("平均掌握", f"{stats['avg_mastery']}%")
    if stats['by_subject']: st.bar_chart(pd.DataFrame(stats['by_subject']).set_index('subject')['total'])
    else: st.info("暂无数据")
