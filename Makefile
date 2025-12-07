# Makefile for less-biased-news project on Ubuntu
# Provides installation, setup, and maintenance tasks

.PHONY: help install check-system install-system-deps setup-venv install-python-deps \
	install-ollama setup-config test run clean uninstall check-ollama pull-ollama-model \
	start-ollama stop-ollama test-ollama

# Variables
PYTHON := python3.12
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
CONFIG_FILE := config.yaml
CONFIG_EXAMPLE := config.yaml.example
OLLAMA_MODEL := llama3.1:8b
OUTPUT_DIR := output

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)Less-Biased News - Ubuntu Installation and Management$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)make install$(NC)          - Complete installation (system deps + Python + Ollama)"
	@echo "  $(YELLOW)make check-system$(NC)     - Check system requirements"
	@echo "  $(YELLOW)make install-system-deps$(NC) - Install Ubuntu system packages"
	@echo "  $(YELLOW)make setup-venv$(NC)       - Create Python virtual environment"
	@echo "  $(YELLOW)make install-python-deps$(NC) - Install Python dependencies"
	@echo "  $(YELLOW)make install-ollama$(NC)   - Install Ollama"
	@echo "  $(YELLOW)make check-ollama$(NC)     - Check if Ollama is installed and running"
	@echo "  $(YELLOW)make pull-ollama-model$(NC) - Pull the default Ollama model"
	@echo "  $(YELLOW)make start-ollama$(NC)     - Start Ollama service"
	@echo "  $(YELLOW)make stop-ollama$(NC)      - Stop Ollama service"
	@echo "  $(YELLOW)make test-ollama$(NC)      - Test Ollama installation"
	@echo "  $(YELLOW)make setup-config$(NC)     - Setup configuration file"
	@echo "  $(YELLOW)make test$(NC)             - Run tests"
	@echo "  $(YELLOW)make run$(NC)              - Run the news filter"
	@echo "  $(YELLOW)make clean$(NC)            - Clean temporary files and cache"
	@echo "  $(YELLOW)make uninstall$(NC)        - Remove virtual environment and dependencies"
	@echo ""

# Complete installation
install: check-system install-system-deps setup-venv install-python-deps install-ollama setup-config
	@echo "$(GREEN)✓ Installation complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Edit $(CONFIG_FILE) with your API keys"
	@echo "  2. Start Ollama: make start-ollama"
	@echo "  3. Pull model: make pull-ollama-model"
	@echo "  4. Test setup: make test"
	@echo "  5. Run application: make run"
	@echo ""

# Check system requirements
check-system:
	@echo "$(BLUE)Checking system requirements...$(NC)"
	@echo -n "  OS: "
	@if [ -f /etc/os-release ]; then \
		. /etc/os-release && echo "$$NAME $$VERSION" ; \
	else \
		echo "$(RED)Unknown$(NC)" ; \
	fi
	@echo -n "  Python 3.12: "
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Found$(NC) ($$($(PYTHON) --version))" ; \
	else \
		echo "$(RED)✗ Not found$(NC)" ; \
		echo "    Install with: sudo apt install python3.12 python3.12-venv python3.12-dev" ; \
		exit 1 ; \
	fi
	@echo -n "  RAM: "
	@total_ram=$$(free -g | awk '/^Mem:/{print $$2}') ; \
	if [ $$total_ram -ge 8 ]; then \
		echo "$(GREEN)✓ $$total_ram GB$(NC)" ; \
	else \
		echo "$(YELLOW)⚠ $$total_ram GB (8GB+ recommended)$(NC)" ; \
	fi
	@echo -n "  CPU cores: "
	@cores=$$(nproc) ; \
	if [ $$cores -ge 4 ]; then \
		echo "$(GREEN)✓ $$cores cores$(NC)" ; \
	else \
		echo "$(YELLOW)⚠ $$cores cores (4+ recommended)$(NC)" ; \
	fi
	@echo -n "  Disk space: "
	@avail=$$(df -BG . | awk 'NR==2 {print $$4}' | sed 's/G//') ; \
	if [ $$avail -ge 20 ]; then \
		echo "$(GREEN)✓ $${avail}GB available$(NC)" ; \
	else \
		echo "$(YELLOW)⚠ $${avail}GB available (20GB+ recommended)$(NC)" ; \
	fi
	@echo "$(GREEN)✓ System requirements check complete$(NC)"

# Install Ubuntu system packages
install-system-deps:
	@echo "$(BLUE)Installing Ubuntu system packages...$(NC)"
	sudo apt update
	sudo apt install -y \
		python3.12 \
		python3.12-venv \
		python3.12-dev \
		build-essential \
		libssl-dev \
		libffi-dev \
		libxml2-dev \
		libxslt1-dev \
		zlib1g-dev \
		libjpeg-dev \
		libpng-dev \
		libopencv-dev \
		python3-opencv \
		git \
		curl \
		wget
	@echo "$(GREEN)✓ System packages installed$(NC)"

# Setup Python virtual environment
setup-venv:
	@echo "$(BLUE)Setting up Python virtual environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV) ; \
		echo "$(GREEN)✓ Virtual environment created$(NC)" ; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(NC)" ; \
	fi

# Install Python dependencies
install-python-deps: setup-venv
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PYTHON_VENV) -m spacy download en_core_web_sm
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"

# Install Ollama
install-ollama:
	@echo "$(BLUE)Installing Ollama...$(NC)"
	@if command -v ollama >/dev/null 2>&1; then \
		echo "$(YELLOW)Ollama already installed$(NC)" ; \
		ollama --version ; \
	else \
		curl -fsSL https://ollama.ai/install.sh | sh ; \
		echo "$(GREEN)✓ Ollama installed$(NC)" ; \
	fi

# Check Ollama installation and status
check-ollama:
	@echo "$(BLUE)Checking Ollama installation...$(NC)"
	@if command -v ollama >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Ollama is installed$(NC)" ; \
		ollama --version ; \
		if curl -s http://localhost:11434 >/dev/null 2>&1; then \
			echo "$(GREEN)✓ Ollama service is running$(NC)" ; \
			echo "  Installed models:" ; \
			ollama list || true ; \
		else \
			echo "$(YELLOW)⚠ Ollama is not running$(NC)" ; \
			echo "  Start with: make start-ollama" ; \
		fi \
	else \
		echo "$(RED)✗ Ollama is not installed$(NC)" ; \
		echo "  Install with: make install-ollama" ; \
	fi

# Pull Ollama model
pull-ollama-model:
	@echo "$(BLUE)Pulling Ollama model: $(OLLAMA_MODEL)$(NC)"
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "$(RED)✗ Ollama is not installed$(NC)" ; \
		echo "  Install with: make install-ollama" ; \
		exit 1 ; \
	fi
	ollama pull $(OLLAMA_MODEL)
	@echo "$(GREEN)✓ Model $(OLLAMA_MODEL) pulled successfully$(NC)"

# Start Ollama service
start-ollama:
	@echo "$(BLUE)Starting Ollama service...$(NC)"
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "$(RED)✗ Ollama is not installed$(NC)" ; \
		exit 1 ; \
	fi
	@if curl -s http://localhost:11434 >/dev/null 2>&1; then \
		echo "$(YELLOW)Ollama is already running$(NC)" ; \
	else \
		nohup ollama serve > ollama.log 2>&1 & \
		echo $$! > ollama.pid || { echo "$(RED)✗ Failed to create PID file$(NC)" ; exit 1 ; } ; \
		sleep 2 ; \
		if curl -s http://localhost:11434 >/dev/null 2>&1; then \
			echo "$(GREEN)✓ Ollama service started (PID: $$(cat ollama.pid))$(NC)" ; \
		else \
			echo "$(RED)✗ Failed to start Ollama$(NC)" ; \
			cat ollama.log ; \
			rm -f ollama.pid ; \
			exit 1 ; \
		fi \
	fi

# Stop Ollama service
stop-ollama:
	@echo "$(BLUE)Stopping Ollama service...$(NC)"
	@if [ -f ollama.pid ]; then \
		pid=$$(cat ollama.pid) ; \
		if [ -n "$$pid" ] && [ "$$pid" -gt 0 ] 2>/dev/null; then \
			if kill -0 $$pid 2>/dev/null; then \
				kill $$pid 2>/dev/null && echo "$(GREEN)✓ Ollama service stopped (PID: $$pid)$(NC)" ; \
			else \
				echo "$(YELLOW)Process $$pid not running$(NC)" ; \
			fi \
		else \
			echo "$(RED)✗ Invalid PID in ollama.pid$(NC)" ; \
		fi ; \
		rm -f ollama.pid ; \
	else \
		echo "$(YELLOW)No Ollama PID file found$(NC)" ; \
	fi

# Test Ollama installation
test-ollama:
	@echo "$(BLUE)Testing Ollama installation...$(NC)"
	@if ! curl -s http://localhost:11434 >/dev/null 2>&1; then \
		echo "$(RED)✗ Ollama is not running$(NC)" ; \
		exit 1 ; \
	fi
	@echo "Testing with simple prompt..." ; \
	response=$$(curl -s http://localhost:11434/api/generate -d '{ \
		"model": "$(OLLAMA_MODEL)", \
		"prompt": "Say hello in one word", \
		"stream": false \
	}') ; \
	if command -v jq >/dev/null 2>&1; then \
		echo "Response: $$(echo "$$response" | jq -r '.response // "No response"')" ; \
	else \
		echo "Response: $$(echo "$$response" | $(PYTHON) -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', 'No response'))")" ; \
	fi ; \
	echo "$(GREEN)✓ Ollama is working$(NC)"

# Setup configuration file
setup-config:
	@echo "$(BLUE)Setting up configuration...$(NC)"
	@if [ ! -f "$(CONFIG_FILE)" ]; then \
		if [ -f "$(CONFIG_EXAMPLE)" ]; then \
			cp $(CONFIG_EXAMPLE) $(CONFIG_FILE) ; \
			echo "$(GREEN)✓ Configuration file created from example$(NC)" ; \
			echo "$(YELLOW)⚠ Please edit $(CONFIG_FILE) with your API keys$(NC)" ; \
		else \
			echo "$(RED)✗ $(CONFIG_EXAMPLE) not found$(NC)" ; \
			exit 1 ; \
		fi \
	else \
		echo "$(YELLOW)Configuration file already exists$(NC)" ; \
	fi
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p cache

# Run tests
test: setup-venv
	@echo "$(BLUE)Running tests...$(NC)"
	@if [ ! -f "$(CONFIG_FILE)" ]; then \
		echo "$(RED)✗ Configuration file not found$(NC)" ; \
		echo "  Run: make setup-config" ; \
		exit 1 ; \
	fi
	$(PYTHON_VENV) test_llm_prompts.py --config $(CONFIG_FILE) --num-articles 5
	@echo "$(GREEN)✓ Tests completed$(NC)"

# Run the news filter
run: setup-venv
	@echo "$(BLUE)Running news filter...$(NC)"
	@if [ ! -f "$(CONFIG_FILE)" ]; then \
		echo "$(RED)✗ Configuration file not found$(NC)" ; \
		echo "  Run: make setup-config" ; \
		exit 1 ; \
	fi
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON_VENV) news_filter.py \
		--config $(CONFIG_FILE) \
		--format html \
		--output "$(OUTPUT_DIR)/news_$$(date +%Y-%m-%d).html" \
		--log INFO
	@echo "$(GREEN)✓ News filter completed$(NC)"
	@echo "  Output: $(OUTPUT_DIR)/news_$$(date +%Y-%m-%d).html"

# Clean temporary files and cache
clean:
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -f ollama.log 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -f ollama.pid 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned temporary files$(NC)"

# Deep clean including cache and logs
clean-all: clean
	@echo "$(BLUE)Cleaning cache, output, and logs...$(NC)"
	rm -rf cache/*
	rm -rf $(OUTPUT_DIR)/*
	find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Deep clean completed$(NC)"

# Uninstall (remove virtual environment)
uninstall: clean
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"
	@echo ""
	@echo "$(YELLOW)Note:$(NC) System packages and Ollama were not removed."
	@echo "To remove Ollama: sudo rm -f /usr/local/bin/ollama"
	@echo "To remove system packages: sudo apt remove [package-name]"
