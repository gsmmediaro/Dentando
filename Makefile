# ─────────────────────────────────────────────────────
# Dental Tooth Caries AI — Makefile
# ─────────────────────────────────────────────────────
PYTHON ?= python
STREAMLIT ?= streamlit
PKG := dental_tooth_caries_ai

.PHONY: setup download prepare train eval demo clean clinical-calibrate clinical-eval clinical-triage release-gates mine-failures reliability-cycle

# ── Setup ──────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r $(PKG)/requirements.txt

# ── Download ───────────────────────────────────────
# Usage: make download DATASET=dentex|bitewing
download:
ifeq ($(DATASET),dentex)
	$(PYTHON) -m $(PKG).datasets.download_dentex
else ifeq ($(DATASET),bitewing)
	$(PYTHON) -m $(PKG).datasets.download_bitewing_mendeley
else
	@echo "Usage: make download DATASET=dentex|bitewing"
	@exit 1
endif

# ── Prepare ────────────────────────────────────────
# Usage: make prepare DATASET=dentex|bitewing
prepare:
ifeq ($(DATASET),dentex)
	$(PYTHON) -m $(PKG).datasets.prepare_dentex
else ifeq ($(DATASET),bitewing)
	$(PYTHON) -m $(PKG).datasets.prepare_bitewing_caries
else
	@echo "Usage: make prepare DATASET=dentex|bitewing"
	@exit 1
endif

# ── Train ──────────────────────────────────────────
# Usage: make train MODALITY=pano|bitewing
EPOCHS ?= 50
IMGSZ  ?= 640
BATCH  ?= 16
train:
	$(PYTHON) -m $(PKG).train --modality $(MODALITY) --task detect \
		--classes caries_only --epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH)

# ── Eval ───────────────────────────────────────────
# Usage: make eval MODALITY=pano|bitewing
eval:
	$(PYTHON) -m $(PKG).eval --modality $(MODALITY)

# ── Demo ───────────────────────────────────────────
demo:
	$(STREAMLIT) run $(PKG)/app.py

# ── Clean ──────────────────────────────────────────
clean:
	rm -rf runs/ results/

# ── Clinical Reliability ───────────────────────────
# Usage examples:
# make clinical-calibrate WEIGHTS=... DATA_YAML=...
# make clinical-eval WEIGHTS=... DATA_YAML=...
# make clinical-triage WEIGHTS=... INPUT=... CLASS_NAMES="Caries,Deep Caries"
# make release-gates EVAL_REPORT=... TRIAGE_REPORT=...
clinical-calibrate:
	$(PYTHON) scripts/calibrate_thresholds.py --weights $(WEIGHTS) --data-yaml $(DATA_YAML) --split $(or $(SPLIT),val) --output $(or $(OUTPUT),results/calibrated_thresholds.json)

clinical-eval:
	$(PYTHON) scripts/evaluate_clinical.py --weights $(WEIGHTS) --data-yaml $(DATA_YAML) --split $(or $(SPLIT),test) --thresholds-json $(or $(THRESHOLDS),results/calibrated_thresholds.json) --critical-classes "$(or $(CRITICAL_CLASSES),)" --output $(or $(OUTPUT),results/clinical_eval_report.json)

clinical-triage:
	$(PYTHON) scripts/triage_with_uncertainty.py --weights $(WEIGHTS) --input $(INPUT) --class-names "$(CLASS_NAMES)" --critical-classes "$(or $(CRITICAL_CLASSES),)" --thresholds-json $(or $(THRESHOLDS),results/calibrated_thresholds.json) --output $(or $(OUTPUT),results/triage_report.json)

release-gates:
	$(PYTHON) scripts/check_release_gates.py --eval-report $(EVAL_REPORT) --triage-report $(TRIAGE_REPORT) --gates $(or $(GATES),configs/release_gates.yml) --output $(or $(OUTPUT),results/release_gate_report.json)

mine-failures:
	$(PYTHON) scripts/mine_failures.py --eval-report $(EVAL_REPORT) --output-dir $(or $(OUTPUT_DIR),results/hard_cases)

reliability-cycle:
	$(PYTHON) scripts/reliability_cycle.py --weights $(WEIGHTS) --data-yaml $(DATA_YAML) --split $(or $(SPLIT),val) --critical-classes "$(or $(CRITICAL_CLASSES),)" --gates $(or $(GATES),configs/release_gates.yml) --output-dir $(or $(OUTPUT_DIR),results/reliability_cycle)
