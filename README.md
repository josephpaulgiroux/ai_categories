# AI Categories Game

A multiplayer word game where AI language models compete to provide unique answers within given categories. Models score points by giving answers that no other AI player provides.

[Read about results from games of AI Categories here](https://artificialcreativity.substack.com/p/a-creativity-test-for-ai-part-1).

## Overview

AI Categories is a turn-based game where multiple AI models compete to give unique responses for different categories (like Animals, Cities, etc.) starting with specific letters. Only unique answers score points, encouraging creative and strategic thinking.

## Prerequisites

- Python 3.7+
- OpenRouter API key (set in `.env` file)

### Required Packages


## Installation

1. Clone the repository:

```
git clone git@github.com:josephpaulgiroux/ai_categories.git
```

2. Install dependencies:

```
pip install openai python-dotenv regex
```

3. Set your OpenRouter API key in the `.env` file:

```
OPENROUTER_API_KEY=your_openrouter_api_key
``` 


## API Requirements

This project uses the OpenRouter API to access various language models. You'll need:
1. An OpenRouter account
2. A valid API key with sufficient credits
3. Access to the models you want to use

Visit [OpenRouter](https://openrouter.ai/) to set up your account and obtain an API key.


## Usage

### Basic Game Setup

Run the game with default settings:

From repo root:
```
python ai_categories.py
```

## Configuration Options

You can configure the game using command-line arguments:

```bash
python ai_categories.py [OPTIONS]
```

### Available Options

- `--blind`: Play in blind mode where models don't see other players' answers (false by default)
- `--feedback`: Provide feedback to the model about their previous turn's response in the game (false by default)
- `--reflect`: Encourage step-by-step thinking in model responses (false by default)
- `--change_mind`: Allow models to change their answers within a single response (false by default)
- `--cheater MODEL_NAME`: Designate a model that will receive information about scoring loopholes (none by default)
- `--starting_point A`: Start with the letter A, or any other letter of the alphabet
- `--keyword STRING`: Custom identifier for the game session (current timestamp by default)
- `--temperature FLOAT`: Set the temperature for model responses (default: 0.0)
- `--turns_back N`: Number of previous turns to include in game history (default: 1)

## Default Game Settings

- **Categories**: Includes 50 diverse categories like Animals, Cities, Foods, etc.

- **Models**: Includes various AI models:
  - Claude 3.5 (Sonnet & Haiku)
  - GPT-4 & GPT-3.5
  - Liquid 3B
  - Mistral 7B
  - Gemini Flash
  - DeepSeek 8B

## Output Files

The game generates three output files:
1. `{keyword}_aicategories_gamenarrative.txt`: Detailed game play-by-play
2. `{keyword}_aicategories_spreadsheet.csv`: Round-by-round responses and scores
3. `{keyword}_aicategories_summary.csv`: Final game summary and scores

## Programmatic Usage

You can also import and use the game in your own Python code:

```python
from ai_categories.ai_categories import AICategoryGame

game = AICategoryGame(
    blind=True,
    reflect=False,
    change_mind=False,
    starting_point=0,
    temperature=0.0
)
game.play()
game.write_results()
```



### Custom Model Configuration

You can pass your own list of models with custom configurations when initializing the game:

```python
custom_models = {
    "Claude 3.5 Sonnet": {
        "model": "anthropic/claude-3.5-sonnet",
        "temperature": 0.7,  # Override default temperature
        "custom_prompt": ""  # Add model-specific prompt additions
    },
    "GPT-4": {
        "model": "openai/gpt-4o",
        "temperature": 0.0,
        "custom_prompt": ""
    }
}

game = AICategoryGame(
    models=custom_models,
    blind=True,
    temperature=0.0
)
```

Each model in the configuration dictionary requires:
- `model`: The OpenRouter model identifier
- `temperature`: Custom temperature setting (or `None` to use game default)
- `custom_prompt`: Additional prompt text specific to this model

