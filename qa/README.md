# QA Validation

Feature QA checklists live here. One file per feature, named by date and feature.

Each checklist is written by Auran (Code channel) after a deploy. Olivia runs each check and pastes the actual terminal output under "Verified by Olivia" — this creates an audit trail showing exactly what was tested and what the production system returned.

## Convention

- **Location**: `qa/` in the repo where the feature deployed
- **Filename**: `YYYYMMDD-feature-name.md`
- **Structure**: Checklist steps with commands → expected output → actual verified output
- **Handoffs**: Separate from QA — those stay in `charting_territory/sessions/code/handoffs/`
- **Historical**: Older QA docs (e.g., from PR24) were moved here from `chat/docs/` for consolidation
