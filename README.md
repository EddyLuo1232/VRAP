<h2 align="center"> <a>⛓‍💥 Computer-Use Agent Frameworks Can Expose Real-World Risks Through Tactics, Techniques, and Procedures</a></h2>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h5>

<h5 align="center">

[Weidi Luo*](https://eddyluo1232.github.io/), [Qiming Zhang](https://scholar.google.com/citations?user=hgu_aPwAAAAJ&hl=en), [Tianyu Lu](https://scholar.google.com/citations?user=kkiCj18AAAAJ&hl=en), [Xiaogeng Liu](https://xiaogeng-liu.com/), [Bin Hu](https://bin-hu.com/), [Hung-Chun CHIU](https://qhjchc.notion.site/), [Siyuan Ma](https://scholar.google.com/citations?user=GSTUhwcAAAAJ&hl=zh-CN)

[Zhen Xiang](https://zhenxianglance.github.io/), [Chaowei Xiao](https://xiaocw11.github.io/)

<p align="center">
  <a href="https://arxiv.org/abs/2504.19373">
  <img src="https://img.shields.io/badge/ArXiv-2504.19373-b31b1b.svg?style=flat-square&logo=arxiv" alt="arXiv">
</a>
  <a href="https://huggingface.co/datasets/DoxxingTeam/DoxBench">
  <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg?style=flat-square&logo=huggingface" alt="Hugging Face">
</a>

</a>

  <img src="https://img.shields.io/badge/Model-Type%3A%20MLRM%20%2F%20MLLM-yellowgreen?style=flat-square">
</p>

## Abstract
Computer-use agent (CUA) frameworks have repeatedly been shown to be vulnerable to jailbreaks, adversarial prompts, and prompt injection. These failures can manipulate CUA frameworks to execute malicious actions or deviate from the user’s intent. However, most evaluations still rely on naive direct, or subjective malicious tasks (e.g., Delete all files or Downloading risk link), which
state only the intended damage while ignoring prerequisites such as permissions, missing services, or environment limits; they also rarely measure what actually happens when the action runs. As a result, such evaluations overestimate the real-world attack capability of CUA frameworks. In practice, adversaries pursue goals through stepwise tactics, techniques, and procedures˜(TTPs): they explore the system, adapt to feedback, and chain small actions into a malicious result. Thus, current studies underestimate the practical threat posed by such realistic attackers. To fill this gap, we present the first systematic study of the realistic threat of CUA frameworks when attackers drive them with TTP. We construct AdvCUA, a dataset of 120 tasks, including 80 TTP-based malicious tasks grounded by MITREE ATT&CK framework and 40 naive direct malicious tasks, spanning 10 tactics and 77 techniques. To measure real-world impact, we design a minimal enterprise micro sandbox with a workstation, an administration server, and a web server, with reproducible checkers that verify task execution and goal completion. Using this setup, we evaluate 3 representative CUA frameworks, including two open-source CUA frameworks (ReAct and AutoGPT) and one closed-source CUA framework (Google Gemini CLI ). We find that current CUA frameworks can complete most end-to-end malicious tasks from TTP-level prompts even without jailbreak strategies, which indicates that current CUA frameworks are vulnerable not only to naive direct malicious requests but also to realistic, TTP-driven adversarial strategies. This underscores that evaluations limited to direct malicious requests substantially underestimate the severe security risks of CUA frameworks in real-world enterprise environments when misused by attackers.


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
