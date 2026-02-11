# ─────────────────────────────────────────────────────
# Dental Tooth Caries AI — Makefile
# ─────────────────────────────────────────────────────
PYTHON ?= python
STREAMLIT ?= streamlit
PKG := dental_tooth_caries_ai

.PHONY: setup download prepare train eval demo clean

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
