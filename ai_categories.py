from util.util import *
from util.prompts import *
import regex as re
from collections import defaultdict, Counter, OrderedDict
import random
import json
from datetime import datetime
import csv
import argparse
from types import SimpleNamespace



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", action="store_true", description="Play in blind mode")
    parser.add_argument("--reflect", action="store_true", description="Think step by step about your answer, but only the text inside the curly brackets will be scored")
    parser.add_argument("--change_mind", action="store_true", description="You can change your answer as many times as you want in your response to attempt to find a unique answer. Only the FINAL response will be scored")
    parser.add_argument("--cheater", type=str, default=None, description="The model that will be given cheating instructions. Other models will be informed that cheating is possible.")
    parser.add_argument("--starting_point", type=int, default=None, description="Where in the alphabet to start the game. Letters will be played in order from this point, wrapping around to the beginning of the alphabet after the end is reached. This allows many combinations of different games, while allowing a single prompt set to be repeated with different configurations.  In this way, you can isolate the impact of those changes on identical game situations")
    parser.add_argument("--keyword", type=str, default=None, description="A string to use to identify this unique run of the game. This keyword will be used in the filenames of output files for easy identification. By default, the current timestamp is used.")
    parser.add_argument("--temperature", type=float, default=0.0, description="The temperature of the model responses")
    parser.add_argument("--turns_back", type=int, default=1, description="The number of turns back to look at for game history when in open mode. Higher values will increase the amount of information available to the model and consume more tokens. Keep in mind that model context window limitations are not automatically checked.")
    return parser.parse_args()

def main():

    args = parse_args()
    game = AICategoryGame(config=args)
    game.play()
    game.write_results()


DEFAULT_MODELS = {
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o-mini": "openai/gpt-4o-mini",
    "GPT-3.5-turbo": "openai/gpt-3.5-turbo",
    "Liquid 3B": "liquid/lfm-3b",
    "Mistral 7B": "mistralai/mistral-7b-instruct-v0.3",
    "Gemini Flash 1.5 8B": "google/gemini-flash-1.5-8b",
    "DeepSeek 8B": "deepseek/deepseek-r1-distill-llama-8b"
}

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class AICategoryGame:

    """
    This is a class that simulates a game of AI Categories. You can import this class and pass in configuration arguments directly to finetune your test.
    """

    def __init__(
            self, 
            models=None, 
            categories=None, 
            blind=True, 
            reflect=False, 
            change_mind=False, 
            starting_point=None, 
            temperature=0.0,
            cheater=None,
            config=None,
        ):
        self.letters = LETTERS[starting_point:] + LETTERS[:starting_point]
        self.models = models or DEFAULT_MODELS
        self.categories = categories or DEFAULT_CATEGORIES
        self.buffer = ""
        self.game_history = []
        self.response_rows = []
        self.personal_histories = defaultdict(str)
        self.personal_responses = defaultdict(str)

        if config is not None:
            self.config = config
        else:
            self.config = SimpleNamespace(
                blind=blind,
                change_mind=change_mind,
                reflect=reflect,
                cheater=cheater,
                turns_back=turns_back,
                temperature=temperature,
                starting_point=starting_point,
            )


        if starting_point is None:
            starting_point = random.randint(0, len(LETTERS) - 1)
        if starting_point < 0 or starting_point >= len(LETTERS):
            raise ValueError(f"Starting point must be between 0 and {len(LETTERS) - 1}")
            


    @property
    def instructions(self):
        return BASE_INSTRUCTIONS

    def get_configuration_instructions(self, model=None):
        output = ""
        if self.config.blind:
            pass
            output += BLIND_MODE_TRUE_INSTRUCTIONS
        else:
            output += BLIND_MODE_FALSE_INSTRUCTIONS

        if self.config.reflect:
            output += REFLECT_INSTRUCTIONS

        if self.config.change_mind:
            output += CHANGE_MIND_TRUE_INSTRUCTIONS
        else:
            output += CHANGE_MIND_FALSE_INSTRUCTIONS
        if self.config.cheater is not None:
            if self.config.cheater == model:
                output += CHEATER_INSTRUCTIONS
            else:
                output += NON_CHEATER_INSTRUCTIONS
        return output



    def get_final_prompt(self, category, letter):
        return FINAL_PROMPT_UNFORMATTED.format(category=category, letter=letter)

    def get_response(self, model_id, category, letter):
        return send_openrouter_request(
            system_prompt=(
                self.instructions 
                + self.get_configuration_instructions(model_id)), 
            content=(
                self.get_content_prompt(model_id)
                + self.get_final_prompt(category, letter)),
            model=self.models[model_id],
            model_id=model_id,
            temperature=self.config.temperature,
        )


    def get_game_history_prompt(self, turns_back=1):
        return  "Pay close attention to the players' responses from the previous round. If you see a pattern, use it to your advantage to score more points.\n\n" + "\n\n".join(self.game_history[-turns_back:])

    def get_personal_history_prompt(self, model_id):
        return (
            "Here was your response last round. Take this into account so that you improve your performance over multiple rounds: " 
            + self.personal_histories[model_id]
            + self.personal_responses[model_id]
        )

    def get_content_prompt(self, model_id):
        return (
               self.get_game_history_prompt(turns_back=self.config.turns_back)
               + self.get_personal_history_prompt(model_id)
        )


    def parse_response(self, response):
        results = re.findall(r"[{]([^{}]+)[}]", response)
        if not results:
            return ""
        if self.config.change_mind:
            return self.clean_response(results[-1])
        else:
            return self.clean_response(results[0])


    def clean_response(self, response):
        return response.replace("{", "").replace("}", "").strip().upper()


    def display_scores(self):
        print("Current scores:")
        for model, score in sorted(
            self.final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True):
            print(f"{model}: {score}")
        print(f"Total: {sum(self.final_scores.values())}")

    def play(self):

        self.buffer = self.instructions
        self.buffer += self.get_configuration_instructions()
        print(self.buffer)
        self.final_scores = Counter()
        for category in self.categories:
            round_results = {}
            letter = self.letters[0]
            self.letters = self.letters[1:] + [letter]

            self.buffer += f"""
=================
{category} ({letter})

"""
            for model_id, model in self.models.items():
                full_response = self.get_response(model_id, category, letter)
                self.personal_responses[model_id] = full_response
                self.buffer += f"""
=================
{model} response:
-----------------
{full_response}

"""
                response = self.parse_response(full_response)
                round_results[model_id] = response

            points, text = self.award_points(round_results)
            self.buffer += text

            result_spreadsheet_row = OrderedDict(category=category, letter=letter)
            for model_id, score in points.items():
                self.final_scores[model_id] += score
                result_spreadsheet_row[model_id] = round_results[model_id]
                result_spreadsheet_row[f"{model_id}_full_response"] = self.personal_responses[model_id]
                result_spreadsheet_row[f"{model_id}_points"] = score

            self.response_rows.append(result_spreadsheet_row)


    def write_results(self):
        keyword = self.config.keyword or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.summarize_results()
        with open(f"{keyword}_aicategories_gamenarrative.txt", "w") as f:
            f.write(json.dumps(self.config.__dict__, indent=4))
            f.write(self.buffer)    
        
        with open(f"{keyword}_aicategories_spreadsheet.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.response_rows[0].keys())
            writer.writeheader()
            writer.writerows(self.response_rows)

        with open(f"{keyword}_aicategories_summary.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_spreadsheet[0].keys())
            writer.writeheader()
            writer.writerows(self.summary_spreadsheet)


    def award_points(self, round_results):
        points = {}
        text = "Previous round results:\n"
        counts = Counter(round_results.values())
        for model_id, response in round_results.items():
            if counts[response] == 1:
                points[model_id] = 1
                msg = f"{model_id} received one point for unique response: {response}\n"
                print(msg)
                text += msg
                self.personal_histories[model_id] = f"Last round YOU ({model_id}) received one point for unique response: {response}\n"
            else:
                points[model_id] = 0
                msg = f"{model_id} received no points for duplicate response: {response}\n"
                print(msg)
                text += msg
                self.personal_histories[model_id] = f"Last round YOU ({model_id}) received no points for duplicate response: {response}\n"

        self.game_history.append(text)
        return points, text


    def summarize_results(self):
        self.buffer += "Final Scores:\n"
        scores_sorted = sorted(self.final_scores.items(), key=lambda x: x[1], reverse=True)
        self.summary_spreadsheet = []
        for model_id, score in scores_sorted:
            self.summary_spreadsheet.append(OrderedDict(model=model_id, score=score))
            self.buffer += f"{model_id}: {score}\n"
        self.summary_spreadsheet.append(OrderedDict(model="Total", score=sum(self.final_scores.values())))
        self.buffer += f"Total: {sum(self.final_scores.values())}\n"



if __name__ == "__main__":
    main()