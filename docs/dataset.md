# HC3 Dataset Documentation
## Human ChatGPT Comparison Corpus (HC3)

### 1. Dataset Provenance and License

| Field | Detail |
|---|---|
| **Dataset Name** | Human ChatGPT Comparison Corpus (HC3) |
| **Official Repository** | [Hello-SimpleAI/HC3 on HuggingFace](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| **Scholarly Citation** | Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.* arXiv preprint arXiv:2301.07597. |
| **License** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) |
| **Acquisition Date** | 2026-10-20 |
| **Language** | English (HC3 also has a Chinese subset `hc3_chinese`, but only the English subset `all` is used in this study) |

---

### 2. Dataset Structure and Schema

HC3 originates from open Q&A and dialogue platforms paired with ChatGPT (specifically `text-davinci-002` / early `gpt-3.5-turbo` era) responses to identical prompts.

#### Raw Data Structure
- `id`: Unique identifier for each question/prompt.
- `question`: The human prompt, question, or discussion seed.
- `human_answers`: A list of one or more human responses collected from Reddit, StackExchange, Wikipedia, or medical forums.
- `chatgpt_answers`: A list of one or more ChatGPT responses generated for the identical prompt.
- `source`: The topic domain category (`reddit_eli5`, `finance`, `open_qa`, `medicine`, `wiki_csai`).

#### Flattened & Processed Schema
For binary classification, nested answer arrays are flattened into one text instance per row:
- `text`: The string of the answer.
- `label`: Integer binary label (`0` = Human-Written, `1` = AI-Generated).
- `label_name`: Human-readable label string (`human` vs. `ai`).
- `source`: Domain/topic source.
- `question`: The original prompt/question context.
- `answer_index`: Index of the response within the original list.
- `char_count`: Character length of the text.
- `word_count`: Whitespace-delimited word count.
- `sentence_count`: Sentence count using regex sentence boundary delimiters.

---

### 3. Programmatically Verified Statistics

The following statistics were directly calculated from the verified dataset:

```
============================================================
HC3 DATASET SUMMARY STATISTICS (Verified)
============================================================
Raw question records          : 24,322
Flattened total answers       : 85,431
Exact duplicates removed      : 6,100
Final usable clean samples    : 79,331

Class Breakdown:
  - Human-written (Label 0)   : 53,086 (66.92%)
  - AI-generated  (Label 1)   : 26,245 (33.08%)
  - Imbalance ratio           : ~ 2.02 : 1 (Human : AI)

Domain Breakdown (Total: 5 domains):
  1. reddit_eli5              : 62,087 (78.26%)
  2. finance                  :  8,393 (10.58%)
  3. open_qa                  :  4,689 ( 5.91%)
  4. medicine                 :  2,549 ( 3.21%)
  5. wiki_csai                :  1,613 ( 2.03%)

Text Length Statistics (Words):
  - Human: Mean = 134.88, Median = 84.0, Std = 164.91, Min = 1, Max = 7,904
  - AI   : Mean = 173.82, Median = 174.0, Std = 60.81,  Min = 1, Max =   639

Text Length Statistics (Characters):
  - Human: Mean = 686.63, Median = 427.0, Std = 833.75, Min = 5, Max = 36,793
  - AI   : Mean = 1015.14, Median = 1015.0, Std = 363.86, Min = 16, Max = 3,508

Outliers:
  - Extremely short (< 10 words): 536 samples
  - Extremely long (> 500 words): 1,815 samples
```

---

### 4. Key Linguistic & Distributional Observations

1. **Length Consistency in AI Text**:
   ChatGPT responses exhibit much tighter variance in length (standard deviation of 60.8 words vs. 164.9 words for human responses). The median AI response is 174 words, while human responses are more skewed with a median of 84 words but long right-tail outliers (up to 7,904 words).
2. **Formatting & Structure**:
   AI text consistently exhibits complete sentences, balanced punctuation, and explanatory tone, whereas human Reddit answers feature casual colloquialisms, contractions, sentence fragments, and variable formatting.
3. **Domain Representation**:
   `reddit_eli5` dominates the sample count (~78%). Domain-stratified evaluation and domain-wise error analysis are therefore essential to ensure detectors do not merely fit Reddit-specific idiosyncrasies.

---

### 5. Data Splitting & Leakage Prevention Protocol

- **Split Ratio**: 70% Train (55,531 samples), 15% Validation (11,900 samples), 15% Test (11,900 samples).
- **Stratification**: Stratified by binary class label.
- **Random Seed**: Fixed `RANDOM_STATE = 42`.
- **Leakage Safeguards**:
  - Exact text duplicates were identified and dropped prior to splitting.
  - TF-IDF vectorizer vocabulary and IDF weights are computed strictly on the training partition.
  - Stylometric feature scalers (mean, variance) are fitted strictly on the training partition.
  - Final held-out test partition is evaluated strictly once per finalized model.
