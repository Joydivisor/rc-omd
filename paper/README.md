# Paper draft

`main.tex` is the working manuscript for the controlled RC-OMD study.

The paper intentionally separates:

1. exploratory development diagnostics;
2. the frozen OOD protocol `ood-v1-2026-08-08`;
3. future function-approximation and language-model work.

The main empirical claim is a Pareto claim about success-learning efficiency and
policy KL at known distractor positions. It is not a claim of causal credit
identification, universal reward improvement, or LLM-scale validation.

Compile from the repository root with the bundled LaTeX compile helper, or use a
standard LaTeX installation:

```powershell
python C:\Users\Administrator\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py `
  E:\RC-OMD\paper\main.tex `
  --output-directory E:\RC-OMD\paper\build
```

If the generated OOD figure exists locally, it is included automatically.
Otherwise the manuscript still compiles with the numerical results table.
