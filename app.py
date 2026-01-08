import streamlit as st
import os
import pandas as pd
from document_processor import DocumentProcessor
from knowledge_base import KnowledgeBase
from feynman_engine import FeynmanEngine
from progress_tracker import ProgressTracker
from config import DOCUMENTS_DIR

st.set_page_config(page_title="费曼 AI 学习助手", page_icon="🧠", layout="wide")

@st.cache_resource
def get_components():
    return {'kb': KnowledgeBase(), 'engine': FeynmanEngine(), 'tracker': ProgressTracker()}

components = get_components()
st.sidebar.title("🧠 费曼 AI (4050版)")
page = st.sidebar.radio("导航", ["📚 导入知识", "✍️ 费曼练习", "📊 进度看板"])

if page == "📚 导入知识":
    st.header("导入本地文档")
    col1, col2 = st.columns([2, 1])
    with col1:
        file = st.file_uploader("支持 PDF/Word/MD", type=['pdf', 'docx', 'md', 'txt'])
        subject = st.text_input("学科/标签", value="通用")
        if file and st.button("开始处理", type="primary"):
            save_path = os.path.join(DOCUMENTS_DIR, file.name)
            with open(save_path, "wb") as f: f.write(file.getbuffer())
            with st.spinner("正在向量化 (利用 4050 CPU)..."):
                count = components['kb'].add_document(save_path, subject)
                st.success(f"成功导入 {count} 个知识块！")
    with col2:
        st.subheader("现有知识库")
        st.write(components['kb'].get_all_subjects() or "暂无数据")

elif page == "✍️ 费曼练习":
    st.header("费曼学习模式")
    if 'session' not in st.session_state: st.session_state.session = None
    if 'eval_result' not in st.session_state: st.session_state.eval_result = None

    col_a, col_b = st.columns([3, 1])
    with col_a:
        subjects = ["全部"] + components['kb'].get_all_subjects()
        sel_subj = st.selectbox("选择复习领域", subjects)
        if st.button("🎯 获取一个概念", type="primary"):
            subj_param = None if sel_subj == "全部" else sel_subj
            res = components['engine'].study_session(subj_param)
            if "error" in res: st.error(res['error'])
            else:
                st.session_state.session = res
                st.session_state.eval_result = None
                st.rerun()
    with col_b:
        due = components['tracker'].get_due_reviews()
        st.metric("今日待复习", len(due))
    st.divider()

    if st.session_state.session:
        data = st.session_state.session
        st.info(f"模式：{data['mode']} | 来源：{data['knowledge']['metadata'].get('source', '未知')}")
        st.markdown(f"### Q: {data['question']}")
        with st.expander("🔍 查看原始知识 (学习完再看)"): st.code(data['knowledge']['content'])
        user_input = st.text_area("你的通俗解释：", height=150)
        if st.button("提交评估") and user_input:
            with st.spinner("AI 老师正在批改..."):
                res = components['engine'].submit_explanation(data['knowledge'], user_input)
                st.session_state.eval_result = res
        if st.session_state.eval_result:
            r = st.session_state.eval_result
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总分", int(r['overall_score']*100))
            c2.metric("准确", int(r['accuracy']*100))
            c3.metric("简洁", int(r['simplicity']*100))
            c4.metric("完整", int(r['completeness']*100))
            st.info(f"💡 点评: {r['feedback']}")
            if r.get('simple_explanation'): st.success(f"参考: {r['simple_explanation']}")

elif page == "📊 进度看板":
    st.header("学习数据统计")
    stats = components['tracker'].get_statistics()
    k1, k2 = st.columns(2)
    k1.metric("知识点总数", stats['total_knowledge'])
    k2.metric("平均掌握度", f"{stats['avg_mastery']}%")
    if stats['by_subject']:
        st.bar_chart(pd.DataFrame(stats['by_subject'], columns=['学科', '数量']).set_index('学科'))
