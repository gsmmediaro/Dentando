# Clinician Review Template (Fast)

Use this for short review batches (20-30 hard cases).

## 1) Open the review CSV

Use the generated file (example):

- `results/reviewed_hard_cases_pano_round1_small30/clinician_review.csv`

Columns to fill:

- `review_status`: `verified` or `uncertain`
- `clinician_action`: `correct`, `wrong_class`, `missed_lesion`, `remove_box`, `uncertain`
- `corrected_label_file`: label filename if corrected
- `notes`: optional short explanation

## 2) Keep decisions simple

- If clearly valid -> `verified` + `correct`
- If label class looks wrong -> `verified` + `wrong_class`
- If missing finding -> `verified` + `missed_lesion`
- If not sure -> `uncertain` + `uncertain`

## 3) What goes into retraining

- Include only rows with `review_status=verified`
- Exclude all `uncertain` rows from retraining
- Keep the original and corrected label files for traceability

## 4) Suggested review target

- 20 to 30 cases per session
- 10 to 20 minutes for a dentist first pass
