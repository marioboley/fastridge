---
name: Student Dataset Structure
status: open — pending UCI site availability for confirmation
---

# Student Dataset Structure

## Summary

The `student` entry in `DATASETS` currently uses `from_ucimlrepo(320)`, which returns only the Portuguese language dataset (649 rows). The UCI entry also contains a math dataset (395 rows). The two files share column names but the grade columns (`G1`, `G2`, `G3`) refer to different subjects and should not be merged into a single target.

## Findings

**Two files, one UCI entry (id=320):**
- `student-por.csv`: 649 students, Portuguese language course grades
- `student-mat.csv`: 395 students, math course grades
- 382 students appear in both files (identified by matching demographics in the original paper: Cortez & Silva 2008)
- `ucimlrepo` returns only the Portuguese file — likely a limitation of their API for multi-file datasets

**Grade columns are course-specific:**
- `G1`, `G2`, `G3` are period 1, period 2, and final grades **for the respective course**
- In student-por: Portuguese language grades; in student-mat: math grades
- Concatenating both files and treating `G3` as a single target mixes two different outcome variables — semantically incorrect

**Legacy behaviour:**
- Legacy experiments used both files concatenated (n≈1044 → n_train≈730) with `G1` and `G2` excluded (p=39)
- This produced inflated R² (~0.83) likely because mixing two subjects creates a noisier target, inadvertently making the regression easier by averaging out subject-specific variance
- Dropping `G1`/`G2` from the Portuguese-only dataset gives R²≈0.27, which is more honest

**Current behaviour:**
- `from_ucimlrepo(320)` → Portuguese only (649 rows, n_train=454)
- `G1` and `G2` included as features (p=41 after one-hot encoding)
- R²≈0.83 — inflated because G1 and G2 are near-duplicates of the target G3

## Proposed Fix

Treat the two files as independent datasets, mirroring the `naval_propulsion` and `parkinsons` pattern:

```python
'student_mat': {'sources': [...]},  # math course, 395 rows
'student_por': {'sources': [...]},  # Portuguese course, 649 rows (replaces 'student')
```

Task specs with `drop=['G1', 'G2']` for both. The 382 students in both files remain in both datasets — they have genuinely different grade sequences for different subjects, so no deduplication is needed.

The source for `student_mat` needs a URL or zip loader since `ucimlrepo(320)` only returns `student_por`. The UCI zip at `https://archive.ics.uci.edu/static/public/320/student+performance.zip` contains both CSVs — but this needs to be verified once the UCI site is back up.

## Open Questions

1. **Confirm grade semantics** — verify from the UCI dataset page or original paper that G1/G2/G3 are indeed course-specific (not a shared grading period across subjects). *Cannot verify while UCI site is down.*
2. **Source for student_mat** — confirm the zip URL and that `student-mat.csv` is available there.
3. **Rename or keep `student`?** — replacing `student` with `student_mat`/`student_por` is a breaking change to any existing task specs referencing `'student'`. The notebook currently uses `'student'` — it would need updating.
