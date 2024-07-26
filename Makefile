.PHONY: build
build: 
	pip install .

clean:
	rm -r ./build/ ./dist/ ./NEoST.egg-info/
