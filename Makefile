.PHONY: build
install:
	python3 -m pip install .

clean:
	rm -rf ./build/ ./dist/ ./NEoST.egg-info/
