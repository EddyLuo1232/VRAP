# OS Agent - Automated Operating System Testing

## Overview
This project contains the OS Agent, an automated system for testing and evaluating attack techniques in containerized environments.

## Project Structure

```
jailbreakS/
├── os_agent.py                 # Main OS Agent script
├── container.py               # Container management utilities
├── utils.py                   # Utility functions and configurations
├── docker-compose.yml         # Docker container configuration
├── Dockerfile                 # Docker image build configuration
├── requirements.txt           # Python dependencies
├── data/                      # Test data and questions
│   ├── test.json             # Default test data
│   ├── test2.json            # Additional test data
│   └── TA*.json              # MITRE ATT&CK technique data
├── OS_Agent_log/             # Output logs and results
└── README.md                 # This file
```

## Core Files

### os_agent.py
The main script that orchestrates the testing process:
- Loads test data from JSON files
- Manages Docker containers
- Executes agent interactions
- Performs verification and saves results

### container.py
Container management class for Docker operations:
- Container lifecycle management
- Command execution in containers
- SSH connectivity handling

### utils.py
Utility functions and configuration classes:
- Agent and Session management
- Configuration parsing
- Helper functions

## Requirements
- Python 3.8+
- Docker and docker-compose
- Dependencies listed in requirements.txt

## Usage

### Basic Usage
```bash
python os_agent.py --input_file data/our_data.json
```

### Advanced Options
```bash
python os_agent.py \
    --input_file data/our_data.json \
    --output_dir OS_Agent_log \
    --agent_model gpt-4o \
    --seed 44
```

## Configuration

### Docker Environment
The system uses three containers:
- `ssh_server` (192.168.2.100) - Target server
- `ssh_client` (192.168.2.101) - Agent execution environment  
- `victim_client` (192.168.2.150) - Additional target (optional, controlled by `local2` config)

### Test Data Format
Test data should include:
- `technique_id`: MITRE ATT&CK technique IDs
- `attacker_identity`: User context for execution
- `evaluation`: Verification configuration
  - `match`: Text patterns to find in output
  - `trigger`: Commands to execute before verification
  - `probe`: Independent verification commands
  - `verify`: Persistent container verification commands
- `configuration`: Environment settings
  - `local2`: Whether to start victim_client container

## Output
Results are saved to the `OS_Agent_log/result/` directory with detailed execution logs and verification outcomes.

## Recent Updates
- ✅ Simplified verification system with trigger/probe/verify methods
- ✅ Clean agent output formatting with emojis
- ✅ Conditional victim_client container startup based on configuration
- ✅ Project cleanup - removed unused agent scripts and data

## Author
This project is part of a security research initiative for automated penetration testing. 
