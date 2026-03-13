PYTHON    := python3
TRACKER   := src/nfold_tracking/TrackingMain.py
OPTITRACK := $(CURDIR)/src/nfold_tracking/optitrack/videos
RESULTS   := $(CURDIR)/src/nfold_tracking/optitrack/results

# ── helpers ───────────────────────────────────────────────────────────────────
run = $(PYTHON) $(TRACKER) --mode $(1) --calib $(2) --config $(2) --video90 $(3)

# ── nfold ─────────────────────────────────────────────────────────────────────
nfold-outdoor-1:
	$(call run,nfold,outdoor,nfold_test1.mp4)

nfold-outdoor-2:
	$(call run,nfold,outdoor,nfold_test2.mp4)

nfold-indoor:
	$(PYTHON) $(TRACKER) --mode nfold --calib indoor --config indoor

# ── aruco ─────────────────────────────────────────────────────────────────────
aruco-outdoor-1:
	$(call run,aruco,outdoor,aruco_test1.mp4)

aruco-outdoor-2:
	$(call run,aruco,outdoor,aruco_test2.mp4)

aruco-indoor:
	$(PYTHON) $(TRACKER) --mode aruco --calib indoor --config indoor

aruco-optitrack-1:
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_1.mp4

aruco-optitrack-2:
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_2.mp4

aruco-optitrack-3:
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_3.mp4

# ── nfold optitrack ───────────────────────────────────────────────────────────
nfold-optitrack-1:
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_1.mp4

nfold-optitrack-2:
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_2.mp4

nfold-optitrack-3:
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_3.mp4

# ── hybrid optitrack ──────────────────────────────────────────────────────────
hybrid-optitrack-1:
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_1.mp4

hybrid-optitrack-2:
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_2.mp4

hybrid-optitrack-3:
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_3.mp4

# ── hybrid ────────────────────────────────────────────────────────────────────
hybrid-outdoor-1:
	$(call run,hybrid,outdoor,hybrid_test1.mp4)

hybrid-outdoor-2:
	$(call run,hybrid,outdoor,hybrid_test2.mp4)

hybrid-indoor:
	$(PYTHON) $(TRACKER) --mode hybrid --calib indoor --config indoor

# ── CSV logging (pose saved to src/nfold_tracking/optitrack/results/) ─────────
aruco-optitrack-1-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_1.mp4 --output $(RESULTS)/aruco_1.csv

aruco-optitrack-2-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_2.mp4 --output $(RESULTS)/aruco_2.csv

aruco-optitrack-3-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode aruco --calib phone --config optitrack --video90 $(OPTITRACK)/Aruco_3.mp4 --output $(RESULTS)/aruco_3.csv

hybrid-optitrack-1-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_1.mp4 --output $(RESULTS)/hybrid_1.csv

hybrid-optitrack-2-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_2.mp4 --output $(RESULTS)/hybrid_2.csv

hybrid-optitrack-3-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode hybrid --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_3.mp4 --output $(RESULTS)/hybrid_3.csv

nfold-optitrack-1-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_1.mp4 --output $(RESULTS)/nfold_1.csv

nfold-optitrack-2-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_2.mp4 --output $(RESULTS)/nfold_2.csv

nfold-optitrack-3-csv: | $(RESULTS)
	$(PYTHON) $(TRACKER) --mode nfold --calib phone --config optitrack --video90 $(OPTITRACK)/Hybrid_3.mp4 --output $(RESULTS)/nfold_3.csv

log-all: aruco-optitrack-1-csv aruco-optitrack-2-csv aruco-optitrack-3-csv \
         hybrid-optitrack-1-csv hybrid-optitrack-2-csv hybrid-optitrack-3-csv \
         nfold-optitrack-1-csv  nfold-optitrack-2-csv  nfold-optitrack-3-csv

$(RESULTS):
	mkdir -p $(RESULTS)

# ── capture ───────────────────────────────────────────────────────────────────
capture:
	$(PYTHON) src/nfold_tracking/VideoGnssCapture.py

capture-no-preview:
	$(PYTHON) src/nfold_tracking/VideoGnssCapture.py --no-preview

# ── help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Tracking targets:"
	@echo "  make nfold-outdoor-1      nfold tracker, outdoor calib, nfold_test1.mp4"
	@echo "  make nfold-outdoor-2      nfold tracker, outdoor calib, nfold_test2.mp4"
	@echo "  make nfold-indoor         nfold tracker, indoor calib, default videos"
	@echo "  make aruco-outdoor-1      aruco tracker, outdoor calib, aruco_test1.mp4"
	@echo "  make aruco-outdoor-2      aruco tracker, outdoor calib, aruco_test2.mp4"
	@echo "  make aruco-indoor         aruco tracker, indoor calib, default videos"
	@echo "  make aruco-optitrack-1    aruco tracker, phone calib, Aruco_1.mp4"
	@echo "  make aruco-optitrack-2    aruco tracker, phone calib, Aruco_2.mp4"
	@echo "  make aruco-optitrack-3    aruco tracker, phone calib, Aruco_3.mp4"
	@echo "  make hybrid-optitrack-1   hybrid tracker, phone calib, Hybrid_1.mp4"
	@echo "  make hybrid-optitrack-2   hybrid tracker, phone calib, Hybrid_2.mp4"
	@echo "  make hybrid-optitrack-3   hybrid tracker, phone calib, Hybrid_3.mp4"
	@echo "  make nfold-optitrack-1    nfold tracker, phone calib, Nfold_1.mp4"
	@echo "  make nfold-optitrack-2    nfold tracker, phone calib, Nfold_2.mp4"
	@echo "  make nfold-optitrack-3    nfold tracker, phone calib, Nfold_3.mp4"
	@echo "  make hybrid-outdoor-1     hybrid tracker, outdoor calib, hybrid_test1.mp4"
	@echo "  make hybrid-outdoor-2     hybrid tracker, outdoor calib, hybrid_test2.mp4"
	@echo "  make hybrid-indoor        hybrid tracker, indoor calib, default videos"
	@echo ""
	@echo "CSV logging targets (output to src/nfold_tracking/optitrack/results/):"
	@echo "  make aruco-optitrack-1-csv   aruco tracker → results/aruco_1.csv"
	@echo "  make aruco-optitrack-2-csv   aruco tracker → results/aruco_2.csv"
	@echo "  make aruco-optitrack-3-csv   aruco tracker → results/aruco_3.csv"
	@echo "  make hybrid-optitrack-1-csv  hybrid tracker → results/hybrid_1.csv"
	@echo "  make hybrid-optitrack-2-csv  hybrid tracker → results/hybrid_2.csv"
	@echo "  make hybrid-optitrack-3-csv  hybrid tracker → results/hybrid_3.csv"
	@echo "  make nfold-optitrack-1-csv   nfold tracker → results/nfold_1.csv"
	@echo "  make nfold-optitrack-2-csv   nfold tracker → results/nfold_2.csv"
	@echo "  make nfold-optitrack-3-csv   nfold tracker → results/nfold_3.csv"
	@echo "  make log-all                 run all 9 recordings and save CSVs"
	@echo ""
	@echo "Capture targets:"
	@echo "  make capture              video + GNSS capture (with preview)"
	@echo "  make capture-no-preview   video + GNSS capture (headless)"

.PHONY: nfold-outdoor-1 nfold-outdoor-2 nfold-indoor \
        nfold-optitrack-1 nfold-optitrack-2 nfold-optitrack-3 \
        aruco-outdoor-1 aruco-outdoor-2 aruco-indoor \
        aruco-optitrack-1 aruco-optitrack-2 aruco-optitrack-3 \
        hybrid-outdoor-1 hybrid-outdoor-2 hybrid-indoor \
        hybrid-optitrack-1 hybrid-optitrack-2 hybrid-optitrack-3 \
        aruco-optitrack-1-csv aruco-optitrack-2-csv aruco-optitrack-3-csv \
        hybrid-optitrack-1-csv hybrid-optitrack-2-csv hybrid-optitrack-3-csv \
        nfold-optitrack-1-csv nfold-optitrack-2-csv nfold-optitrack-3-csv \
        log-all capture capture-no-preview help
