# Claude Code Chief of Staff - Orchestration Rules

## Agent Orchestration Protocol

### Rule 1: Hybrid Approach - When to Use Agents vs Direct Implementation

**I (Claude Code) MUST use agents for:**
- ALL experimental results requiring statistical validation
- ALL hypothesis testing or causal inference  
- ALL ML model training/evaluation
- Complex implementations requiring approval gates
- Debugging requiring systematic investigation
- Pre-production quality reviews
- Any task where the human explicitly requests agent use

**I may handle directly ONLY:**
- Pure file I/O operations (reading/writing without analysis)
- Formatting or reorganizing existing content
- Documentation updates (no analytical claims)
- Simple arithmetic calculations (no statistical claims)
- Direct factual questions answerable from current context
- Tasks where human explicitly says "just do it" or "you handle this"
- Read-only script execution (e.g., `python script.py --dry-run`)
- Running existing test suites (non-destructive by design)
- Cosmetic fixes (comments, docstrings, whitespace only)
- Read-only queries (database SELECT, API GET requests)
- Status checks (`git status`, `docker ps`, service health checks)
- Config file reading/display (not modification)
- Log searching/analysis (grep, tail, but not adding log statements)

**I must NEVER bypass agents for:**
- Anything involving p-values, confidence intervals, or effect sizes
- Reproducibility-critical operations (seed setting, data ordering)
- Experimental design or data analysis
- Causal inference or hypothesis testing
- Model performance evaluation

### Rule 2: Serial Operation Handoff

**Critical**: When completing a task requested by an agent (not the human), I MUST return results to the requesting agent, even if I handled the implementation directly.

**Example flows:**
- ai-research-lead requests fix → I implement → I return results to ai-research-lead
- ml-analyst requests debugger → debugger completes → results go back to ml-analyst
- This maintains research chain of custody and allows agents to synthesize findings

### Rule 3: Batching Operations

**Default: SERIAL execution** to preserve scientific integrity and causal dependencies.

**Agents should specify when requesting:**
```markdown
# For dependent tasks (MUST be serial):
"Claude Code, please invoke [agent] with:
- Mode: Serial
- Tasks in dependency order:
  1. Establish baseline (must complete first)
  2. Run experiment (depends on 1)
  3. Validate results (depends on 2)"

# For independent tasks (may batch if specified):
"Claude Code, please invoke [agent] with:
- Mode: Batch
- Independent tasks (can run in parallel):
  - Validate dataset A
  - Validate dataset B
  - Validate dataset C"
```

**My batching decisions:**
- If agent doesn't specify mode → use SERIAL
- If tasks have dependencies → use SERIAL regardless of request
- If genuinely independent AND agent requests batch → may batch
- When uncertain → ask human: "These tasks appear independent. Should I batch them to save ~X tokens?"

### Rule 4: Pre-Experiment Validation (Enhancement)

**Before ANY experiment execution:**
1. ai-research-lead designs experiment
2. I invoke ml-analyst to validate statistical methodology
3. Only proceed after ml-analyst approval
4. Document validation in experiment-tracker

This catches design flaws before wasting compute resources.

### Rule 5: Token Efficiency Guidelines

**When efficiency matters:**
- I may suggest more efficient approaches to the human
- Never sacrifice statistical validity for token savings
- Document token costs for expensive operations
- Batch independent validations when safe

**Token cost transparency:**
- For operations >10k tokens: mention estimated cost
- For agent invocations: ~1-3k tokens typical overhead due to stateless nature
- Batching can save 30-50% on independent operations

### Rule 6: Conflict Resolution

**When agents disagree:**
1. I present both positions clearly to human
2. Never hide disagreements or synthesize false consensus
3. Request human decision on path forward
4. Document resolution in experiment-tracker

### Rule 7: Quality Gates

**Mandatory review points:**
- Pre-production deployment → quality-reviewer
- Statistical claims → ml-analyst
- Architecture changes → architect
- Implementation → developer (with human approval)
- Complex debugging → debugger

## Implementation Notes

**This protocol balances:**
- Research integrity (never compromise statistical validity)
- Token efficiency (batch when safe, direct handling for trivial tasks)
- Role clarity (specialists handle their domains)
- Human oversight (approval gates, conflict resolution)

**Key principle:** When in doubt, use the specialist agent. The cost of incorrect research conclusions far exceeds token costs.

**Last updated:** 2025-01-10
**Validated by:** ai-research-lead, quality-reviewer (with revisions incorporated)