# AdvCUA - Advanced Container-based Understanding Agent

Automated operating system testing framework using Large Language Models for security assessments in containerized environments.

## Overview

Multi-agent system supporting AutoGPT, React, and MITRE ATT&CK frameworks for penetration testing in isolated Docker containers.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env_template .env
# Edit .env with your API keys

# Build and start containers
docker build -t my-ssh-image .
docker-compose up -d
```

## Usage

```bash
# Basic usage
python vrap.py --dataset data/test_case.json

# Advanced options, Results saved to `OS_Agent_log/` directory.
python vrap.py --framework react --provider openai --model gpt-4
```

### Arguments

- `--dataset`: Test data file path
- `--framework`: Agent type (`autogpt`, `react`, `mitre`)
- `--provider`: LLM provider (`openai`, `openrouter`, `my_api`)
- `--model`: Model name
- `--output_dir`: Results directory

## Test Data Format

```json
{
  "attacker_identity": "user",
  "prompt_to_os_agent": "Task description",
  "evaluation": {
    "match": ["success pattern"],
    "trigger": "setup command",
    "probe": "verification command"
  }
}
```

## Docker Network

- **SSH Server(administration server)** (`192.168.2.100`): Target server
- **SSH Client(work station)** (`192.168.2.101`): Agent environment
- **Victim Client(web server)** (`192.168.2.150`): Optional target
