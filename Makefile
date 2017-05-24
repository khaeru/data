all:

import-chip: chip/cache/2007.log
	python3 chip.py 1995 2002 2007

test:
	py.test-3 chip.py

.PHONY: import-chip test
