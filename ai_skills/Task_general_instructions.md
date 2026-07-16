When generating any task instruction document for this repository, please follow the repository-specific AI workflow convention below.

This repository uses `ai_skills/` as the central directory for AI-assisted task planning, HPC execution records, summaries, and final reports.

Use the following directory structure:

```text
ai_skills/
├── tasks/       # Task instruction documents generated locally
├── logs/        # Raw or lightly cleaned execution logs from HPC / Codex CLI
├── summaries/   # Concise execution summaries written after running tasks
└── reports/     # Final analysis reports synthesized from task results
```

For every new task, generate a task instruction file under:

```text
ai_skills/tasks/
```

Use a dated and descriptive filename, for example:

```text
ai_skills/tasks/2026-06-19_distance_unrolled_eval.md
```

Each task instruction must explicitly tell the HPC-side Codex CLI to save its outputs in the following locations:

```text
ai_skills/logs/
ai_skills/summaries/
```

The expected HPC-side outputs should include:

1. A raw execution log:

```text
ai_skills/logs/YYYY-MM-DD_<task_name>_execution_log.md
```

2. A concise execution summary:

```text
ai_skills/summaries/YYYY-MM-DD_<task_name>_summary.md
```

The execution summary should include:

* task name
* execution date
* machine / hostname if available
* current git branch
* git status before running
* commands executed
* files inspected
* files modified, if any
* outputs generated
* errors or warnings encountered
* whether the task completed successfully
* recommended next steps

If the task produces a higher-level scientific or technical interpretation, ask Codex to save a draft report under:

```text
ai_skills/reports/YYYY-MM-DD_<task_name>_report.md
```

Important rules:

* Do not save task-related logs, summaries, or reports directly under `outputs/`, `predictions/`, `checkpoints/`, or `test_data/`.
* Do not modify or overwrite files in `checkpoints/`, `outputs/`, `predictions/`, or `test_data/` unless the task instruction explicitly allows it.
* Keep large generated data outside git unless explicitly requested.
* Prefer committing only task instructions, lightweight logs, summaries, reports, configs, and small diagnostic figures.
* Every task should be traceable through this chain:

```text
ai_skills/tasks/<task>.md
    ↓
ai_skills/logs/<task>_execution_log.md
    ↓
ai_skills/summaries/<task>_summary.md
    ↓
ai_skills/reports/<task>_report.md
```

When writing the task instruction, include an “Expected output files” section that names the exact log and summary files that should be created by the HPC-side Codex CLI.
