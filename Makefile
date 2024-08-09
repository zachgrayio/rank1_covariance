PROJECT = rank1_covariance
BUILD_DIR = cmake-build-release
GENERATOR = Ninja

run: build run_python
	@echo "CUDA OUTPUT (SHOULD MATCH PYTHON):"
	./$(BUILD_DIR)/$(PROJECT)

build: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake --build .

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -G $(GENERATOR) -DCMAKE_BUILD_TYPE=Release ..

clean:
	rm -rf $(BUILD_DIR)

run_python:
	@echo "PYTHON OUTPUT:"
	python3 ./main.py


.PHONY: run build clean
