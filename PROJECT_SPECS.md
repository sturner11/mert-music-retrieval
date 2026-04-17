# Final Project Specs Tracker

## Submission Deliverable
- Submit **one PDF writeup**.
- Maximum length: **3 pages total**.
- Must include:
  - **Project writeup** (problem, data, approach, results): **1-2 pages** (up to 2 pages).
  - **Time log accounting** (daily breakdown): **up to 1 page**.

## Grading Rubric
- Final project = **25%** of overall course grade.
- Breakdown:
  - **20%** = hours spent.
  - **5%** = report quality.
- Hours scoring formula:
  - Percentage for hours = `total_hours / 30`.
- Report quality depends on:
  - Project quality.
  - Writing quality.

## Core Project Requirement
- Must be a **substantial deep learning project** of your own choosing.
- Must include **training or fine-tuning a model**.
- Not sufficient by itself:
  - Only running inference with a pretrained model.
  - Only calling external APIs (OpenAI, Gemini, etc.).

## Current Project Direction (Locked)
- Primary approach: **fine-tune a MERT encoder for music retrieval embeddings**.
- Similarity target: **performance + sound** (not full composition modeling in phase 1).
- Minimum compliance condition:
  - Must include actual model optimization work (head training and/or encoder fine-tuning).
  - Inference-only use of MERT does not satisfy requirements.
- Baseline comparison:
  - At least one simple baseline (e.g., MFCC + nearest neighbors).
- Evaluation:
  - Primary metric: Recall@10.
  - Secondary metrics: Recall@1, Recall@5, qualitative listening checks.

## Time Log Requirements
- Time must be documented **daily**.
- Each entry must include:
  - Date.
  - Hours.
  - Brief activity description.
- If time is not logged, it does **not** count.
- Example format:
  - `8/11 - 1 hour - read alphago paper`
  - `8/12 - 2 hours - downloaded and cleaned data`

## Hour Constraints
- Maximum countable hours:
  - **Research/reading:** up to **5 hours**.
  - **Prep work:** up to **10 hours**.
    - Examples: data prep/cleaning, setup friction, simulator/model environment setup.
- Minimum required:
  - **At least 20 hours** must be core ML work:
    - Designing, building, debugging, testing deep learning models.
    - Analyzing results.
    - Running experiments/iterations.
- Important:
  - You **may not** count model runtime/training wall-clock time.
  - No extra credit beyond **30 total hours**.

## Report Requirements (1-2 pages)
- Describe clearly:
  - Problem you set out to solve.
  - Exploratory data analysis (EDA).
  - Technical approach.
  - Results.

## Recommended Report Content Checklist
- Dataset discussion:
  - Source/publisher.
  - Why this data matters.
- Problem framing:
  - Classification vs regression.
  - Supervised vs unsupervised.
  - Relevant domain background.
  - Prior approaches and outcomes.
- Data exploration:
  - What data includes.
  - Patterns observed.
  - Relevant visualizations.
- Technical approach:
  - Method background.
  - Model architecture/topology.
  - Training/inference algorithm.
  - Train/test split strategy.
  - Parameter count.
  - Optimizer choice.
  - Pretrained weights (if used) and source.
- Results analysis:
  - Final metric(s) on your split (e.g., RMSE where applicable).
  - Overfitting analysis and evidence.
  - Iterations from initial to final model and why.
  - Progress toward original project goal.

## Project Direction Guidance
- Effort is weighted more than perfect results.
- Better to attempt an ambitious project than a safe/simple one.
- Your time log should clearly communicate scope and effort.
- Acceptable project patterns:
  - Novel idea + novel dataset.
  - Vanilla DNN on novel dataset with strong experimental science.
  - Reimplementation of state-of-the-art method (clear with instructor first).

## Possible Dataset Sources
- Figure Eight (CrowdFlower)
- KDD Cup
- UCI Repository
- CVonline
- Kaggle (current/past)
- Data.gov
- AWS datasets
- World Bank
- BYU CS478 datasets
- data.utah.gov
- Google Research
- BYU DSC competition
- Self-collected dataset

## Project Compliance Quick Check
- [ ] Uses deep learning.
- [ ] Includes training or fine-tuning.
- [ ] Has daily time log entries.
- [ ] Meets hour category limits (research/prep/core work).
- [ ] Shows at least 20 hours of core model work.
- [ ] Report is polished and high quality.
- [ ] PDF is 3 pages max.
- [ ] Writeup section is 1-2 pages.
- [ ] Time accounting section is <= 1 page.
