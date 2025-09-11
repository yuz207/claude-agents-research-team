# Hypothesis Dictionary
*Reference file for all hypothesis definitions - maintained across stateless agent invocations*

## How to Use This File
1. **Agents**: Read this file to look up hypothesis IDs and definitions
2. **Adding New**: Create new ID (H001, H002...) when novel hypothesis proposed
3. **Status Updates**: Change status based on evidence from analyses
4. **Evolution**: Use H001.1, H001.2 for refinements; new number for major changes

## Design Philosophy
This file is kept intentionally FLAT and SIMPLE for efficient grep operations.
- No nested tree structures (those are built on-demand from analyses)
- No complex relationships (just "see also" references)
- Evolution history is derived from chronological analysis reading, not stored here

## Status Definitions
- **PROPOSED**: Hypothesis suggested but not yet tested
- **TESTING**: Currently under investigation
- **CONFIRMED**: Strong evidence supports (p<0.01, effect size>0.5)
- **REJECTED**: Evidence contradicts hypothesis
- **REFINED**: Evolved into new version (see evolution)

---

## Active Hypotheses

### H001: [Example - Linear Decay Optimization]
- **Definition**: Linear learning rate decay maintains training stability while improving convergence
- **Research Question**: What is the optimal learning rate schedule?
- **First Proposed**: [Date] (analysis_XXX)
- **Status**: PROPOSED
- **Current Evidence**: None yet
- **Related Analyses**: []
- **Notes**: [Any additional context]

---

## Archived Hypotheses
*Moved here when CONFIRMED, REJECTED, or superseded*

---

## Evolution History
*Track how hypotheses evolved*

Example:
- H001 → H001.1: Refined to focus on transformer models specifically
- H001.1 → H001.2: Added constraint about batch size interaction