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

profile_ncu: build
	$(eval TIMESTAMP := $(shell date +%Y-%m-%d_%H-%M-%S))
	sudo PROFILE=1 /usr/local/cuda/bin/ncu --nvtx --set full --target-processes all --launch-count 40 --kill no -o /tmp/rank1_covariance.$(TIMESTAMP).ncu-rep ./$(BUILD_DIR)/$(PROJECT)
	ncu-ui /tmp/rank1_covariance.$(TIMESTAMP).ncu-rep

profile_nsys: build
	$(eval TIMESTAMP := $(shell date +%Y-%m-%d_%H-%M-%S))
	PROFILE=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --cudabacktrace=true -x true -o /tmp/rank1_covariance.$(TIMESTAMP).nsys-rep ./$(BUILD_DIR)/$(PROJECT)
	ncu-ui /tmp/rank1_covariance.$(TIMESTAMP).nsys-rep

.PHONY: run build clean profile_ncu
