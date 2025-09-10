# AI Research Team Guide

## Team Overview

Your AI research team consists of specialized agents working under your direction, with Claude Code as your primary interface. This guide documents each team member, their roles, interactions, and real-world workflows.

### Team Structure
```yaml
research_team:
  core_scientists:
    - ai-research-lead      # PI: Designs, experiments, analyzes
    - ml-analyst           # Validates, profiles, optimizes
    - experiment-tracker   # Documents everything
  research_engineers:     # Called when needed for implementation
    - architect           # Designs complex system architectures
    - developer          # Implements new architectures cleanly
    - debugger           # Diagnoses training failures
    - quality-reviewer   # Pre-production validation
  potential_future_members:
    - technical-writer     # Manuscript preparation, literature review
    - science-critic      # Devil's advocate, challenges assumptions
```

```yaml
core_research_team:
   ai-research-lead:
      does_everything:
         - Architecture design
         - Training experiments
         - Data handling (no separate data-engineer needed)
         - Compute management (no separate ml-engineer needed)
         - Initial analysis
   ml-analyst:
      validates_everything:
         - Statistical rigor
         - Empirical verification
         - Benchmark comparisons
   experiment-tracker:
      documents_everything:
         - Experiments
         - Decisions
         - Context
```

## Command Structure

```
YOU (Human CTO/Director)
    ↓ [Direct all decisions]
    Claude Code (Your Executive Interface)
        ↓ [Coordinates]
        ai-research-lead (Principal Investigator)
            ↓ [Delegates to]
            ├── ml-analyst (Senior Empirical Analyst)
            ├── experiment-tracker (Research Secretary)
            └── [When needed] Research Engineers
                ├── architect (Complex designs)
                ├── developer (Clean implementations)
                └── debugger (Failure diagnosis)
```

## Team Members

### 🧑‍💼 **Claude Code** (Your Executive Interface)
**Role**: Your direct interface to the entire research team  
**Quick Summary**: Translates your natural language requests into agent actions, maintains context, ensures nothing happens without your approval

**Key Responsibilities**:
- Takes your instructions and coordinates appropriate agents
- Presents findings back for your approval
- Handles general tasks outside research scope
- Maintains conversation context

**Example Output**:
```
YOU: "Let's improve our model's inference speed"

Claude Code: "I'll have the ai-research-lead analyze optimization opportunities."
[Invokes ai-research-lead]
[Presents findings]
"The research team has identified three approaches. Which would you like to pursue?"
```

---

### 🔬 **ai-research-lead** (Principal Investigator & Lead Data Scientist)
**Role**: Player-coach leading research while doing hands-on data science  
**Quick Summary**: Your senior scientist who both conducts primary analyses and coordinates the research team

**Key Responsibilities**:
- Generates and tests hypotheses
- Performs statistical analyses and modeling
- Coordinates specialist agents
- Makes research recommendations (subject to your approval)
- Synthesizes findings from all team members

**Interfaces With**:
- **Reports to**: You (via Claude Code)
- **Delegates to**: All specialist agents
- **Collaborates with**: ml-analyst (for validation)

**Example Output**:
```
ai-research-lead: "Based on my analysis and team findings:
- Hypothesis: INT8 quantization will maintain accuracy
- Evidence: Literature shows <1% loss for similar architectures
- Team input: ML-analyst confirms statistical viability
- Recommendation: Test with calibration dataset
AWAITING YOUR APPROVAL to proceed with implementation."
```

---

### 📊 **ml-analyst** (Senior ML Performance Analyst)
**Role**: Empirical analyst providing data-driven validation  
**Quick Summary**: Your senior analyst who rigorously tests everything and provides empirical evidence for all claims

**Key Responsibilities**:
- Empirical model evaluation
- Statistical validation
- Root cause analysis
- A/B test analysis
- Production monitoring

**Interfaces With**:
- **Reports to**: ai-research-lead
- **Challenges**: ai-research-lead's hypotheses with data
- **Provides data to**: experiment-tracker for documentation

**Example Output**:
```
ml-analyst: "EMPIRICAL FINDING: Model accuracy drops 15% on sequences >512 tokens
- Statistical significance: p<0.001, effect size d=1.2
- Pattern: Attention weights saturate at position 509-512
- Sample size: N=10,000 test cases
- Confidence interval: [14.2%, 15.8%]
Evidence attached: [graphs, statistical tests, data tables]"
```

---

### 📝 **experiment-tracker** (Research Secretary)
**Role**: Meticulous documenter and record keeper  
**Quick Summary**: Your research librarian who records everything but analyzes nothing

**Key Responsibilities**:
- Documents all experiments
- Takes meeting minutes
- Preserves context before compaction
- Organizes research artifacts
- Maintains searchable records

**Interfaces With**:
- **Documents for**: Everyone
- **Polls**: All agents for milestone summaries
- **Reports to**: ai-research-lead

**Example Output**:
```markdown
# Research Session: 2024-01-15
## Experiments Logged
- exp_047: Mixed precision → 28% speedup, -0.1% accuracy
## Key Decisions
- ✅ Human approved INT8 quantization testing
## Conflicts Recorded
- ML-analyst vs AI-lead on position 509 vs 512
- Resolution: Human directed testing both
## Action Items
- [ ] AI-lead: Design calibration experiment
```

---

### 🏗️ **architect** (Research Engineer - System Design)
**Role**: Designs complex implementations for novel architectures  
**Quick Summary**: Called when research needs sophisticated system design

**Key Responsibilities**:
- Design specifications for novel architectures
- Complex system integration patterns
- Performance architecture for training systems
- Never implements, only designs

**When Called**:
- New architecture needs detailed specification
- Complex multi-component systems
- Distributed training architecture

---

### 💻 **developer** (Research Engineer - Implementation)
**Role**: Clean implementation of research ideas  
**Quick Summary**: Implements novel architectures with production-quality code

**Key Responsibilities**:
- Implement new layers/modules/architectures
- Write comprehensive tests
- Ensure code quality and documentation
- Follow specifications precisely

**When Called**:
- New architecture needs implementation
- Complex modifications to existing code
- Need clean, reproducible implementation

---

### 🔍 **debugger** (Research Engineer - Diagnostics)
**Role**: Diagnoses training failures and mysterious behaviors  
**Quick Summary**: Systematic investigation of model/training issues

**Key Responsibilities**:
- Root cause analysis of training failures
- Investigate NaN losses, gradient explosions
- Debug convergence issues
- Evidence-based diagnosis only

**When Called**:
- Training mysteriously fails
- Model exhibits unexpected behavior
- Need deep investigation of numerical issues

---

### 📚 **technical-writer** (Potential Future Member)
**Role**: Technical documentation and literature synthesis  
**Quick Summary**: Would transform research into manuscripts and conduct literature reviews

**Potential Responsibilities**:
- Write research papers and technical reports
- Conduct systematic literature reviews
- Prepare grant proposals and documentation
- Transform experiment logs into publication-ready content

**Note**: Currently not active. Experiment-tracker handles raw documentation.

---

### 🔬 **science-critic** (Potential Future Member)
**Role**: Devil's advocate and methodological skeptic  
**Quick Summary**: Would challenge assumptions and identify blind spots

**Potential Responsibilities**:
- Challenge experimental designs for bias
- Identify confounding variables
- Question causal assumptions
- Suggest alternative explanations
- Prevent groupthink

**Note**: Currently not active. Skepticism role partially covered by ml-analyst's empirical rigor.

---

## Interaction Workflows

### Weekly Research Workflow

```python
# Monday - Planning Session
YOU: "This week we need to improve inference speed by 50%"

ai-research-lead: "I'll coordinate the team to explore optimization strategies."
[Delegates to ml-analyst for baseline analysis]
[Delegates to literature-researcher for recent papers]

# Tuesday - Hypothesis Generation
ai-research-lead: "Team synthesis complete. Three hypotheses:
H1: Quantization (confidence: high)
H2: Pruning (confidence: medium)  
H3: Distillation (confidence: low)
Recommendation: Start with H1"

YOU: "Approved. Also test mixed precision as baseline."

# Wednesday - Experimentation
ml-analyst: "Baseline established: Mixed precision gives 28% speedup"
ai-research-lead: "Proceeding to quantization testing per approval"
experiment-tracker: "Experiment exp_047 logged successfully"

# Thursday - Results & Conflicts
ml-analyst: "Quantization fails at position 509, not 512 as hypothesized"
ai-research-lead: "Theory suggests 512. Requesting additional analysis"

[CONFLICT ESCALATION TO YOU]
Claude Code: "Disagreement: ML-analyst says position 509 (empirical evidence), 
             AI-lead says 512 (theoretical). Your decision?"

YOU: "Test both positions. The empirical evidence is concerning."

# Friday - Synthesis & Decision
ai-research-lead: "Final results: Position 509 confirmed as issue.
Solution: Adjusted quantization range for attention layers.
Result: 47% speedup, 1.1% accuracy loss
RECOMMENDATION: Deploy to staging"

YOU: "Approved for staging. Monitor for 48 hours before production."

experiment-tracker: "Week's work documented. 5 experiments, 2 conflicts resolved, 
                    1 major discovery (position 509), 1 deployment approved"
```

### Conflict Resolution Scenario

```python
# Disagreement Emerges
ml-analyst: "Data shows 23% failure rate on night images, p<0.001"
ai-research-lead: "I hypothesize insufficient training data"

ml-analyst: "Disagree. Error pattern suggests preprocessing issue.
            Evidence: Failures correlate with gamma correction values"

# Escalation to Human
ai-research-lead: "CONFLICT: Cannot reach consensus with ml-analyst.
ML-analyst: Preprocessing issue (gamma correction)
My position: Training data issue
Both have merit. Requesting human decision."

YOU: "ML-analyst is likely correct. Check preprocessing pipeline first."

# Resolution
ai-research-lead: "Acknowledged. Redirecting investigation to preprocessing."
ml-analyst: "Analyzing preprocessing pipeline as directed..."
experiment-tracker: "Conflict recorded: Preprocessing vs training data.
                    Resolution: Human directed preprocessing investigation."
```

### Daily Standup Structure

```python
# Every Morning
ai-research-lead: "Daily Research Standup:

COMPLETED YESTERDAY:
- 3 experiments on quantization approaches
- Identified position 509 issue

IN PROGRESS TODAY:
- Calibration dataset preparation (ml-engineer)
- Statistical validation of findings (ml-analyst)
- Documentation update (experiment-tracker)

BLOCKERS:
- Need GPU allocation for large-scale test (awaiting human approval)
- Conflicting findings on optimal batch size

DECISIONS NEEDED FROM HUMAN:
1. Approve $2000 compute budget for scaling tests?
2. Priority: accuracy or speed for edge cases?"

YOU: "1. Approved. 2. Prioritize accuracy for medical images, speed for others."
```

### Experiment Execution Flow

```python
# 1. Hypothesis Approval
ai-research-lead: "Hypothesis: Attention pruning will maintain accuracy"
YOU: "Test it, but preserve top-2 heads per layer minimum"

# 2. Experiment Setup
ai-research-lead: "Configuring experiment with your constraints..."
ml-engineer: "Environment prepared, compute allocated"
data-engineer: "Test datasets ready"

# 3. Execution
ai-research-lead: "Running experiment exp_048..."
ml-analyst: "Monitoring metrics in real-time..."

# 4. Analysis
ml-analyst: "Results: 8% accuracy drop, fails requirements"
ai-research-lead: "Analyzing failure modes..."

# 5. Documentation
experiment-tracker: "Experiment exp_048 complete:
- Hypothesis: Attention pruning
- Result: Failed (8% accuracy loss)
- Key learning: Minimum 4 heads needed per layer
- Decision: Hypothesis rejected"

# 6. Next Steps
ai-research-lead: "Based on failure, recommending alternative approach..."
```

## Communication Patterns

### Information Flow
```
Discovery → Analysis → Synthesis → Decision → Implementation
   ↓           ↓          ↓           ↓            ↓
ml-analyst  ml-analyst  ai-lead      YOU      ml-engineer
                          ↑                        ↑
                  experiment-tracker (documents all stages)
```

### Escalation Triggers
1. **Automatic Escalation to You**:
   - Agent disagreements
   - Budget >$1000
   - Production deployments
   - Unexpected findings
   - Failed experiments

2. **Optional Check-ins**:
   - Daily standups
   - Weekly summaries
   - Milestone reviews

### Decision Authority

| Decision Type | Authority | Example |
|--------------|-----------|---------|
| Hypotheses | YOU approve | "Test quantization" → "Yes, but add calibration" |
| Experiments | YOU approve | "Run 100 GPU hours" → "Start with 10 hours pilot" |
| Interpretations | YOU arbitrate | "AI-lead says X, ML-analyst says Y" → "Test both" |
| Deployments | YOU approve | "Deploy to production?" → "Staging first, then production" |
| Resource allocation | YOU approve | "Need $5000 compute" → "Approved, but track usage" |

## Key Principles

### For You (Human CTO)
- You have **final say** on everything
- You can **intervene** at any time
- You get **notified** of important events
- You can **override** any agent conclusion

### For Agents
- **No autonomous execution** without your approval
- **Document everything** via experiment-tracker
- **Escalate conflicts** to you immediately
- **Provide evidence** for all claims

### For the Team
- **Healthy disagreement** improves outcomes
- **Empirical evidence** trumps speculation
- **Reproducibility** is non-negotiable
- **Your decision** is final

## Quick Reference

### Starting a Research Session
```
YOU: "Let's improve [specific metric]"
→ Claude Code: "I'll engage the research team"
→ ai-research-lead: "Coordinating analysis..."
→ [Team works]
→ ai-research-lead: "Here are findings and recommendations"
→ YOU: "Approved/Modified/Rejected"
```

### Checking Progress
```
YOU: "What's the status?"
→ experiment-tracker: "X experiments run, Y decisions made, Z in progress"
→ ai-research-lead: "Current focus: [topic], blocking on: [issue]"
→ ml-analyst: "Recent finding: [empirical result]"
```

### Making Decisions
```
Team: "AWAITING DECISION: [options presented]"
→ YOU: "Go with option A, but modify X"
→ Team: "Acknowledged, proceeding with modified option A"
→ experiment-tracker: "Decision recorded"
```

---

*This document serves as your operational guide to the AI research team. The team provides the analytical and implementation capabilities of a full research lab while maintaining your complete strategic and tactical control.*