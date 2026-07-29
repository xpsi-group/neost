.PHONY: build
install:
	python3 -m pip install .

editable:
	python3 -m pip install -e .

test:
	pytest

clean:
	rm -rf ./build/ ./dist/ ./NEoST.egg-info/
