# Academic Report Data & Reference Guide
## PRG4206 Individual Assignment: A Comparative and Explainable Machine Learning System for AI-Generated Text Detection

This document provides structured, publication-grade academic material covering sections A through Z required for the assignment report.

---

### A. Introduction
The emergence of large language models (LLMs) such as OpenAI's ChatGPT, GPT-4, and Anthropic's Claude represents an inflection point in computational linguistics. While these generative systems exhibit remarkable fluency and utility across education, software development, and technical writing, they concurrently pose significant challenges to academic integrity, digital forensics, intellectual property attribution, and misinformation mitigation. The core problem addressed in this study is the automated, explainable distinction between human-written and AI-generated textual content.

### B. Problem Statement
Generative language models produce grammatically coherent, topic-relevant text with high lexical diversity, making manual detection by human readers unreliable. Furthermore, existing commercial AI detectors operate as closed proprietary black boxes, frequently suffer from high false-positive rates on non-native English writing, and offer zero explainability. There is an urgent academic need for an open, rigorously benchmarked, and explainable machine learning system evaluated on scholarly benchmark datasets.

### C. Research Questions
- **RQ1**: How accurately can classical and modern machine learning models distinguish human-written from AI-generated text using the HC3 benchmark?
- **RQ2**: What performance differentials exist between linear n-gram baselines (Logistic Regression, Linear SVM), tree-based stylometric ensembles (XGBoost), and deep contextual transformers (DistilBERT)?
- **RQ3**: Which surface-level, syntactic, and stylometric characteristics contribute most decisively to separating human and AI text?
- **RQ4**: How robust is detector performance across heterogeneous domains (medical, financial, open dialogue, encyclopedic)?
- **RQ5**: What error patterns (false positives vs. false negatives) characterize the models, and what are their ethical ramifications?

### D. Objectives
1. Acquire and verify the official peer-reviewed HC3 dataset under open licenses.
2. Formulate a rigorous exploratory data analysis characterizing distributional differentials between human and AI writing.
3. Design leakage-free NLP preprocessing and feature extraction pipelines (TF-IDF vs. 21-feature stylometric matrix).
4. Implement, cross-validate, and hyperparameter-tune at least four distinct algorithms: Logistic Regression, Linear SVM, XGBoost, and DistilBERT.
5. Quantitatively benchmark models using eight standard metrics (Accuracy, Precision, Recall, $F_1$, ROC-AUC, PR-AUC, MCC, Specificity).
6. Integrate Explainable AI (XAI) using SHAP and LIME to generate human-interpretable feature attributions.
7. Conduct deep error analysis by domain, text length, and error category.
8. Deliver an interactive Streamlit prototype equipped with live inference, feature attribution, and comparative analytics.

### E. Literature Review
The literature review is grounded in 11 scholarly studies cataloged in `docs/literature_review_table.csv`, including:
- **Guo et al. (2023)** introduced HC3 and established initial RoBERTa detection baselines (~0.95 $F_1$), demonstrating that ChatGPT outputs are more neutral, polite, and grammatically complete than human text.
- **Mitchell et al. (2023)** proposed *DetectGPT*, a zero-shot perturbation-based detector operating on model negative log-likelihood curvature.
- **Sanh et al. (2019)** introduced *DistilBERT*, reducing BERT-base parameters by 40% while retaining 97% language understanding capability.
- **Lundberg & Lee (2017)** formulated *SHAP*, unifying Shapley game-theoretic values for exact local and global model explanations.
- **Ribeiro et al. (2016)** established *LIME* for local surrogate model interpretability.

### F. Research Gap
While prior literature frequently reports near-perfect classification on isolated datasets, existing studies rarely provide:
1. Direct side-by-side comparison of vocabulary-based models (TF-IDF) versus content-agnostic stylometric models (surface metrics) on the same corpus.
2. Systematic domain-stratified breakdown revealing vulnerability to domain shifts.
3. Transparent, open-source local explainability coupled with calibrated confidence bounds.

### G. Dataset Description
The Human ChatGPT Comparison Corpus (HC3) comprises 24,322 question prompts paired with human and ChatGPT answers across 5 domains (`reddit_eli5`, `finance`, `open_qa`, `medicine`, `wiki_csai`). Following flattening and deduplication of 6,100 identical texts, the final verified dataset contains **79,331 unique samples** (53,086 human, 26,245 AI).

### H. Exploratory Data Analysis (EDA)
EDA demonstrated marked structural divergences:
- **Length Regularity**: ChatGPT answers have an extremely tight normal distribution around a median of 174 words (std = 60.8), whereas human responses exhibit high skewness (median = 84 words, std = 164.9, max = 7,904 words).
- **Outliers**: 536 texts contained fewer than 10 words; 1,815 texts exceeded 500 words.
- **Domain Dominance**: `reddit_eli5` represents 78.26% of all texts, necessitating domain-wise stratification.

### I. System Architecture
```
User Text Input
      ↓
[Input Validation & Length Check]
      ↓
┌───────────────────────┴───────────────────────┐
│ Cleaned Representation (Unicode, HTML/URLs)   │ Stylometric Representation (Casing, Punct.)
▼                                               ▼
[TF-IDF N-Gram Vectorizer]                     [21-Dimensional Stylometric Extractor + Scaler]
│                                               │
├─► Logistic Regression (C=10, liblinear)       └─► XGBoost Classifier
├─► Linear SVM (Platt Calibrated)
└─► DistilBERT Tokenizer & Transformer Head
      │
      ▼
[Ensemble / Model Inference] ──► Class Label & Calibrated Confidence
      │
      ▼
[Explainability Engine (SHAP / LIME / Coefficients)]
      │
      ▼
[Streamlit Interactive UI with Warnings & Metrics Dashboard]
```

### J. Preprocessing & Leakage Controls
- Text cleaning strips disruptive zero-width characters and HTML while deliberately preserving commas, colons, and question marks.
- The 70/15/15 train/val/test split was performed with fixed stratification (`random_state=42`).
- All vectorizer vocabularies, IDF weights, and standard scalers were fitted solely on the 55,531 training records.

### K. Feature Engineering
The stylometric pipeline extracts 21 linguistic metrics:
1. Character count
2. Word count
3. Sentence count
4. Average word length
5. Average sentence length
6. Type-token ratio (TTR)
7. Vocabulary richness (Guiraud's R index)
8. Unique word ratio
9. Overall punctuation frequency
10. Comma frequency ratio
11. Period frequency ratio
12. Question mark ratio
13. Exclamation mark ratio
14. Uppercase character ratio
15. Numeric digit ratio
16. Whitespace ratio
17. Stop-word ratio
18. Repeated word frequency
19. Function word ratio
20. Average paragraph length
21. Paragraph count

### L. Model Development & Tuning
- **Logistic Regression**: Evaluated across $C \in \{0.01, 0.1, 1, 10, 100\}$, balanced vs. unweighted classes, and solvers.
- **Linear SVM**: Evaluated across regularization strengths $C$, calibrated via Platt sigmoid scaling.
- **XGBoost**: Evaluated over 30 randomized iterations optimizing tree depth, learning rate, and subsampling.
- **DistilBERT**: Contextual bidirectional transformer fine-tuned for sequence classification with AdamW.

### M. Statistical Testing Protocol & Verified Hypothesis Testing
To rigorously evaluate whether observed performance differences between models are statistically significant, **McNemar's test was applied to paired predictions on the identical held-out test set** ($N = 11,900$, 7,963 Human, 3,937 AI). McNemar's test specifically evaluates marginal homogeneity on paired nominal data by examining discordant classification errors ($b$: Model A correct, Model B wrong; $c$: Model A wrong, Model B correct).

Both the continuity-corrected asymptotic chi-square statistic:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
and the two-sided exact binomial test ($p = 2 \times \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c}$) were computed at significance level $\alpha = 0.05$.

#### Verified Pairwise Statistical Test Results:
1. **Linear SVM vs. Logistic Regression**:
   - Both Correct ($b_{00}$): 11,711
   - Linear SVM only correct ($b_{01} = b$): 24
   - Logistic Regression only correct ($b_{10} = c$): 14
   - Both Wrong ($b_{11}$): 151
   - Total Discordant ($b + c$): 38
   - Continuity-corrected $\chi^2$: **2.1316** ($p = 0.1443$)
   - Exact Binomial $p$-value: **0.1433**
   - **Conclusion**: Fail to reject $H_0$ ($p > 0.05$). There is **no statistically significant difference** in prediction accuracy between Linear SVM and Logistic Regression. Both linear TF-IDF architectures demonstrate functionally equivalent discriminative capability on the HC3 corpus.

2. **Linear SVM vs. XGBoost (Stylometric)**:
   - Both Correct ($b_{00}$): 11,389
   - Linear SVM only correct ($b_{01} = b$): 346
   - XGBoost only correct ($b_{10} = c$): 148
   - Both Wrong ($b_{11}$): 17
   - Total Discordant ($b + c$): 494
   - Continuity-corrected $\chi^2$: **78.5607** ($p = 7.77 \times 10^{-19} < 0.0001$)
   - Exact Binomial $p$-value: **$2.55 \times 10^{-19}$**
   - **Conclusion**: Reject $H_0$ ($p < 0.0001$). The performance advantage of Linear SVM over XGBoost is **statistically significant** ($\Delta \text{Acc} = +1.66\%$, $\Delta F_1 = +2.46\%$).

3. **Logistic Regression vs. XGBoost (Stylometric)**:
   - Both Correct ($b_{00}$): 11,380
   - Logistic Regression only correct ($b_{01} = b$): 345
   - XGBoost only correct ($b_{10} = c$): 157
   - Both Wrong ($b_{11}$): 18
   - Total Discordant ($b + c$): 502
   - Continuity-corrected $\chi^2$: **69.6594** ($p = 7.03 \times 10^{-17} < 0.0001$)
   - Exact Binomial $p$-value: **$3.00 \times 10^{-17}$**
   - **Conclusion**: Reject $H_0$ ($p < 0.0001$). The performance advantage of Logistic Regression over XGBoost is **statistically significant** ($\Delta \text{Acc} = +1.58\%$, $\Delta F_1 = +2.34\%$).

> [!NOTE]
> **Audit Note on Prior Statistical Discrepancy**:
> An earlier iteration of the statistical script erroneously passed the full $2 \times 2$ contingency table `[[b00, b01], [b10, b11]]` to `scipy.stats.chi2_contingency`, which tests for cross-tabular independence rather than paired marginal homogeneity. Because both models are highly accurate, their decisions are strongly dependent, which generated an anomalous Pearson statistic of $\chi^2 = 9299.89$. Independent verification via `src/verify_statistics.py` corrected this to the true McNemar test statistic ($\chi^2 = 2.1316, p = 0.1443$), accurately reflecting that the two linear models do not differ significantly.

### N. Ethical Considerations & Responsible AI
- **Risk of False Accusations**: Academic penalties or professional dismissal based on automated detection represent acute harm. Systems must never be used as sole determinative arbiters.
- **Non-Native English Vulnerability**: Non-native speakers naturally exhibit lower lexical perplexity and simpler grammatical structures, risking disproportionate false positive rates.
- **Model Evasion**: Paraphrasers (e.g., QuillBot) and adversarial prompt modifications easily degrade detector accuracy.
- **Privacy**: The prototype operates entirely locally without transmitting user texts to third-party APIs.

### O. Future Work
1. Incorporating perplexity and burstiness signals computed via open-source LLMs.
2. Adversarial robustness stress-testing against automated paraphrasing tools.
3. Multi-class attribution identifying specific generative architectures (GPT-4 vs. Claude vs. Gemini vs. Llama).

### P. Document Analysis Pipeline & Prototype UI
To transition from isolated paragraph scoring to practical academic file screening, the system incorporates a complete document analysis pipeline (`src/document_parser.py`):
1. **Multi-Format Ingestion**:
   - `.txt`: UTF-8 with automatic fallback decoding (`latin-1`, `cp1252`).
   - `.pdf`: Multi-page text stream extraction using `pypdf`, preserving page indexes.
   - `.docx`: Paragraph-by-paragraph XML DOM extraction via `python-docx`.
2. **Context-Aware Segmentation & Chunking**:
   - Preserves natural paragraph and sentence boundaries.
   - Merges fragmented paragraphs below `min_chunk_words=50` to maintain TF-IDF feature density.
   - Partitions oversized passages exceeding `max_chunk_words=250` along terminal punctuation boundaries.
3. **Document-Level Aggregation**:
   - Computes word-weighted class posterior probability:
     $$P_{\text{doc}}(\text{AI}) = \frac{\sum_{i=1}^K P(y_i = \text{AI}) \cdot w_i}{\sum_{i=1}^K w_i}$$
     where $w_i$ denotes the word count of chunk $i$.
   - Tracks AI chunk percentage, mean chunk confidence, and extracts the top 3 most AI-like and top 3 most Human-like passages for human review.
4. **Interactive Dashboard**:
   - Includes real-time sequential chunk probability timelines, donut classification charts, and sortable prediction tables with CSV export.
   - Clearly disclaims all results as model approximations rather than conclusive authorship determinations.
