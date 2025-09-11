# Experiment Tracking System Guide

## Overview
This system tracks analyses of experiments YOU run on your server. Agents analyze your uploaded logs and document findings.

## Key Files

### 1. `hypothesis_dictionary.md`
- Central reference for all hypotheses
- Agents read this to find hypothesis IDs
- YOU update status based on evidence

### 2. `analyses_index.csv`
- Quick searchable index of all analyses
- Fields: id, date, run_id, type, context, research_question, hypothesis_ref, key_finding, effect_size, priority, checkpoint_ref
- Agents grep this first before reading full analyses

### 3. `current_session.md`
- Active working memory during session
- Resets after compaction

### 4. `AUTOSAVE.md`
- Updates every 10% context usage
- Preserves state for session recovery

### 5. Individual analyses (`by_date/YYYY-MM-DD/analysis_XXX.md`)
- Detailed analysis records
- Created during checkpoint extraction at 80% context

## Workflow

### When You Upload Logs
1. ai-research-lead analyzes your logs
2. Checks hypothesis_dictionary.md for relevant hypotheses
3. ml-analyst validates statistics
4. Findings accumulate in context

### At 80% Context
1. experiment-tracker creates checkpoint
2. Extracts individual analyses from checkpoint
3. Updates analyses_index.csv
4. You approve compaction

### Finding Past Results
```bash
# Quick search in CSV
Grep("quantization", "experiments/analyses_index.csv")

# Get hypothesis details
Grep("H001", "experiments/hypothesis_dictionary.md")

# Read specific analysis
Read("experiments/by_date/2024-01-15/analysis_001.md")
```

## Agent Responsibilities

- **ai-research-lead**: Analyzes logs, proposes hypotheses
- **ml-analyst**: Validates statistics, recommends status changes
- **experiment-tracker**: Documents everything, creates checkpoints
- **You**: Run experiments, update hypothesis status, make decisions