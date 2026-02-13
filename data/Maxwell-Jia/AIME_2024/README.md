---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - explanation-generation
pretty_name: AIME 2024 Dataset
size_categories:
  - n<1K
dataset_info:
  config_name: default
data_files:
  - split: train
    path: aime_2024_problems.parquet
---

# AIME 2024 Dataset

## Dataset Description

This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2024. AIME is a prestigious high school mathematics competition known for its challenging mathematical problems.

## Dataset Details

- **Format**: JSONL
- **Size**: 30 records
- **Source**: AIME 2024 I & II
- **Language**: English

### Data Fields

Each record contains the following fields:
- `ID`: Problem identifier (e.g., "2024-I-1" represents Problem 1 from 2024 Contest I)
- `Problem`: Problem statement
- `Solution`: Detailed solution process
- `Answer`: Final numerical answer

## Purpose

This dataset is primarily used for:
1. Evaluating Large Language Models' (LLMs) mathematical reasoning capabilities
2. Testing models' problem-solving abilities on complex mathematical problems
3. Researching AI performance on structured mathematical tasks

## Features

- Covers various mathematical domains (geometry, algebra, number theory, etc.)
- Includes detailed solution processes for each problem
- All problems have specific numerical answers
- High difficulty level, suitable for testing advanced reasoning capabilities
- Problems require multi-step reasoning and mathematical insight

## Dataset Structure

The dataset is organized in JSONL format, where each line represents a complete problem with its solution. Example:

```json
{
"ID": "2024-I-1",
"Problem": "Problem statement...",
"Solution": "Detailed solution...",
"Answer": "Numerical answer"
}
```
