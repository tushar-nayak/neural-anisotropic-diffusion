PYTHON ?= python

.PHONY: run smoke clean

run:
	$(PYTHON) main.py

smoke:
	$(PYTHON) main.py --epochs 1 --batch-size 16 --neighbor-mode 4 --iterations 1 --lambda-param 0.1 --noise-type gaussian --no-refinement --no-unet-guidance

clean:
	rm -f results/unified_loss_curves.png results/unified_qualitative_results.png results/unified_comparison_table.csv
