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
    * Python 3.11
    * Ollama (已安装 Qwen2.5:3b，nomic-embed-text)
    拉取方法：
    ollama pull qwen2:1.5b 
    ollama pull nomic-embed-text（嵌入模型，用于知识库）
    * NVIDIA GPU (推荐，用于加速 Ollama)

2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
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

* `feynman_engine.
````py`: 系统的“大脑”，负责生成问题和评估答案。
* `knowledge_base.py`: 负责向量
````化存储和检索（ChromaDB）。
* `progress_tracker.py`: 负责记录学习日志
````和计算遗忘曲线（SQLite）。

在上述基础上，按以下要求更改：
1.导入文件后结合概括文件主要内容和涉及领域
2.生成问题和给出解释时以对应领域的专家的思维，辅导学生学习的场景来进行，给出的解释要符合费曼学习法要求，且依托导入的文件知识
3.分知识点时要将对应知识点概括。体现知识点的主要内容
4.参考解释位置要与所涉及知识点符合
5.切换页面卡顿明显，点击学习后卡顿且不会跳转对应知识点的学习
给出修改逻辑和更改后的完整代码

pip install streamlit pdfplumber pandas
