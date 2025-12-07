# Makefile Documentation

This document provides detailed information about using the Makefile for Ubuntu installation and management of the less-biased-news project.

## Quick Start

```bash
# Check if your system meets requirements
make check-system

# Install everything (system packages, Python dependencies, Ollama)
make install

# Start Ollama and pull the model
make start-ollama
make pull-ollama-model

# Test the installation
make test

# Run the news filter
make run
```

## Available Targets

### Installation Targets

#### `make install`
Complete installation including system dependencies, Python packages, and Ollama.

**What it does:**
1. Checks system requirements
2. Installs Ubuntu system packages
3. Creates Python virtual environment
4. Installs Python dependencies
5. Installs Ollama
6. Creates configuration file

**Usage:**
```bash
make install
```

#### `make check-system`
Verify that your system meets the requirements.

**Checks:**
- Operating system version
- Python 3.12 availability
- Available RAM (8GB+ recommended)
- CPU cores (4+ recommended)
- Disk space (20GB+ recommended)

**Usage:**
```bash
make check-system
```

#### `make install-system-deps`
Install Ubuntu system packages required for the project.

**Installs:**
- Python 3.12 and development packages
- Build tools (gcc, make, etc.)
- OpenCV and image processing libraries
- SSL, XML, and other system libraries

**Usage:**
```bash
sudo make install-system-deps
```

**Note:** Requires sudo privileges.

#### `make setup-venv`
Create a Python 3.12 virtual environment.

**Usage:**
```bash
make setup-venv
```

#### `make install-python-deps`
Install all Python dependencies from requirements.txt and download spaCy model.

**Usage:**
```bash
make install-python-deps
```

### Ollama Targets

#### `make install-ollama`
Install Ollama using the official installation script.

**Usage:**
```bash
make install-ollama
```

#### `make check-ollama`
Check if Ollama is installed and running.

**Shows:**
- Ollama version
- Service status (running/not running)
- Installed models

**Usage:**
```bash
make check-ollama
```

#### `make pull-ollama-model`
Pull the default Ollama model (llama3.1:8b).

**Customize the model:**
```bash
# Pull a different model
OLLAMA_MODEL=mistral:7b make pull-ollama-model
```

**Usage:**
```bash
make pull-ollama-model
```

#### `make start-ollama`
Start the Ollama service in the background.

**Creates:**
- `ollama.log` - Service log file
- `ollama.pid` - Process ID file

**Usage:**
```bash
make start-ollama
```

#### `make stop-ollama`
Stop the Ollama service.

**Usage:**
```bash
make stop-ollama
```

#### `make test-ollama`
Test that Ollama is working by sending a simple prompt.

**Usage:**
```bash
make test-ollama
```

### Configuration and Testing

#### `make setup-config`
Copy config.yaml.example to config.yaml.

**Usage:**
```bash
make setup-config
```

**Note:** You'll need to edit config.yaml with your API keys after running this.

#### `make test`
Run the test suite with 5 articles.

**Requirements:**
- Configuration file must exist
- Virtual environment must be set up
- Dependencies must be installed

**Usage:**
```bash
make test
```

### Running the Application

#### `make run`
Run the news filter with default settings.

**What it does:**
1. Creates output directory if needed
2. Runs news_filter.py with HTML output
3. Saves to `output/news_YYYY-MM-DD.html`

**Usage:**
```bash
make run
```

### Maintenance Targets

#### `make clean`
Remove temporary files and caches.

**Removes:**
- Python cache files (__pycache__, *.pyc)
- Log files
- pytest cache
- Ollama PID file

**Usage:**
```bash
make clean
```

#### `make clean-all`
Deep clean including cache and output directories.

**Usage:**
```bash
make clean-all
```

#### `make uninstall`
Remove the virtual environment.

**Usage:**
```bash
make uninstall
```

**Note:** Does not remove system packages or Ollama. See the output for manual removal instructions.

## Configuration Options

### Change Python Version

Edit the Makefile and change:
```makefile
PYTHON := python3.12
```

### Change Ollama Model

Edit the Makefile and change:
```makefile
OLLAMA_MODEL := llama3.1:8b
```

Or set it when running:
```bash
OLLAMA_MODEL=mistral make pull-ollama-model
```

### Change Output Directory

Edit the Makefile and change:
```makefile
OUTPUT_DIR := output
```

## Troubleshooting

### Python 3.12 not found

**Error:** `Python 3.12: ✗ Not found`

**Solution:**
```bash
sudo apt install python3.12 python3.12-venv python3.12-dev
```

### Ollama installation fails

**Error:** Installation script fails

**Solutions:**
1. Check internet connection
2. Try manual installation: Visit https://ollama.ai/download
3. Check system compatibility

### Virtual environment creation fails

**Error:** Cannot create virtual environment

**Solutions:**
```bash
sudo apt install python3.12-venv
```

### Permission denied errors

**Error:** Permission issues when installing system packages

**Solution:**
Use sudo for system-level targets:
```bash
sudo make install-system-deps
```

### Ollama won't start

**Error:** Ollama service fails to start

**Solutions:**
1. Check if port 11434 is already in use:
   ```bash
   lsof -i :11434
   ```
2. Check the log file:
   ```bash
   cat ollama.log
   ```
3. Try manual start:
   ```bash
   ollama serve
   ```

### Model download is slow

**Issue:** Large model downloads take time

**Solutions:**
1. Use a smaller model:
   ```bash
   OLLAMA_MODEL=mistral:7b make pull-ollama-model
   ```
2. Be patient - models are 4-8GB
3. Check internet connection speed

## Tips and Best Practices

### Regular Maintenance

```bash
# Clean up weekly
make clean

# Update Python dependencies
make install-python-deps

# Pull latest model updates
make pull-ollama-model
```

### Development Workflow

```bash
# Initial setup
make install
make start-ollama
make pull-ollama-model

# Daily work
make test          # Test your changes
make run           # Generate news
make clean         # Clean up
```

### Multiple Environments

If you need multiple Python environments:

```bash
# Rename venv
mv venv venv-dev

# Create new one
VENV=venv-prod make setup-venv
```

### Checking Dependencies

```bash
# Verify system packages
dpkg -l | grep python3.12

# Verify Python packages
./venv/bin/pip list

# Verify Ollama models
ollama list
```

## Integration with Other Tools

### Cron Jobs

Run news filter daily at 7 AM:

```bash
# Add to crontab (crontab -e)
0 7 * * * cd /path/to/less-biased-news && make run >> /var/log/news-filter.log 2>&1
```

### Systemd Service

Create a systemd service for Ollama:

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=your-user
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable it:
```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

## Advanced Usage

### Custom Installation Directory

```bash
# Install to custom location
git clone https://github.com/jabraham/less-biased-news.git /opt/news-filter
cd /opt/news-filter
make install
```

### Running with Different Config

```bash
# Use custom config
./venv/bin/python news_filter.py --config my-config.yaml --format html --output output/news.html
```

### Parallel Testing

```bash
# Run more comprehensive tests
./venv/bin/python test_llm_prompts.py --num-articles 50 --fetch
```

## See Also

- [README.md](README.md) - Main project documentation
- [OLLAMA_SETUP.md](OLLAMA_SETUP.md) - Detailed Ollama configuration guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical implementation details

## Support

For issues with:
- **Makefile**: Check this documentation and README.md
- **Ollama**: See OLLAMA_SETUP.md
- **Python errors**: Check requirements.txt and Python version
- **System packages**: See install-system-deps target details
