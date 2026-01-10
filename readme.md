# 费曼 AI 学习助手 (Feynman AI Assistant)

这是一个基于费曼学习法（The Feynman Technique）的智能化学习系统。它不仅仅是阅读工具，更是一个强制你输出、解释、被评估的“AI 教练”。

## 🌟 核心特性 (v2.0 优化版)

1.  **循序渐进模式 (Structured Learning)**：
    * 专为初学者设计。系统会解析文档结构，允许用户按章节顺序选择知识点进行学习，建立完整的知识框架。
2.  **科学评分体系 (Scientific Evaluation)**：
    * **5维评分**：从事实准确性、逻辑连贯性、通俗性（ELI5）、完整性、误区规避五个维度打分。
    * **费曼式参考答案**：AI 会生成一个“给8岁孩子听”的完美解释作为参考。
3.  **智能复习 (Smart Review)**：
    * 内置艾宾浩斯遗忘曲线算法，自动安排复习时间。
4.  **分区数据看板 (Subject Analytics)**：
    * 提供按学科/标签分类的掌握度报表，一眼看清弱项。

## 🛠️ 安装指南

1.  **环境要求**：
    * Python 3.8+
    * Ollama (已安装 Qwen2.5:3b 或其他模型)
    * NVIDIA GPU`markdown
# 费曼 AI 学习助手 (Feynman AI Assistant)

这是一个基于费曼学习法（The Feynman Technique）的智能化学习系统。它不仅仅是阅读工具，更是一个强制你输出、解释、被评估的“AI 教练”。

## 🌟 核心特性 (v2.0 优化版)

1.  **循序渐进模式 (Structured Learning)**：
    * 专为初学者设计。系统会解析文档结构，允许用户按章节顺序选择知识点进行学习，建立完整的知识框架。
2.  **科学评分体系 (Scientific Evaluation)**：
    * **5维评分**：从事实准确性、逻辑连贯性、通俗性（ELI5）、完整性、误区规避五个维度打分。
    * **费曼式参考答案**：AI 会生成一个“给8岁孩子听”的完美解释作为参考。
3.  **智能复习 (Smart Review)**：
    * 内置艾宾浩斯遗忘曲线算法，自动安排复习时间。
4.  **分区数据看板 (Subject Analytics)**：
    * 提供按学科/标签分类的掌握度报表，一眼看清弱项。

## 🛠️ 安装指南

1.  **环境要求**：
    * Python 3.8+
    * Ollama (已安装 Qwen2.5:3b 或其他模型)
    * NVIDIA GPU (推荐，用于加速 Ollama)

2.  **安装依赖**：
    ```bash
    pip install streamlit
```` langchain-ollama langchain-chroma sentence-transformers PyMuPDF python-docx pandas
    ```

3.  **启动
````系统**：
    确保 Ollama 已在后台运行：
    ```bash
    ollama serve
````    ```
    运行应用：
    ```bash
    streamlit run app.py
    ```

## 🚀 使用
````流程

1.  **导入知识**：在“知识库管理”页上传 PDF/Word/MD 文件，
````并打上学科标签（如 "Python基础"）。
2.  **开始学习**：
    * **
````新手**：进入“循序渐进”模式，选择学科，按顺序点击知识点。
    * **
````老手**：进入“随机/复习”模式，让系统随机抽查或推送待复习内容
````。
3.  **费曼练习**：
    * 系统提出引导性问题。
    * 你
````用自己的大白话（设想教给别人）输入解释。
    * 点击提交，查看评分
````、点评及 AI 生成的“标准费曼解释”。
4.  **复盘**：在“学习看板”
````查看各学科的掌握进度条。

## 🧩 目录结构说明

* `feynman_engine.
````py`: 系统的“大脑”，负责生成问题和评估答案。
* `knowledge_base.py`: 负责向量
````化存储和检索（ChromaDB）。
* `progress_tracker.py`: 负责记录学习日志
````和计算遗忘曲线（SQLite）。
