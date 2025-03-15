

import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter
from random import choice
import aioconsole

from google import genai

load_dotenv()






def send_openrouter_request(
    system_prompt, 
    content, 
    temperature=0.0, 
    model="google/gemini-2.0-flash-lite-preview-02-05:free", 
    model_id=None,
    ):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        http_client=None,
    )

    model_id = model_id or model

    messages =[
        {
        "role": "system",
        "content": system_prompt,
        },
        {
        "role": "user",
        "content": content,
        },
    ]

    # for anthropic models, cache system prompt and give a new user message, since this mirrors how the game loop works -- the
    # general instructions are the same, but game history and prompt change every time.
    if "anthropic" in model:
        messages[0]["cache_control"] = {"type": "ephemeral"}


    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "artificialcreativity.substack.com", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "Artificial Creativity", # Optional. Site title for rankings on openrouter.ai.
            },
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt. Waiting for response.")
        input("Press Enter to continue...")
        return "No response returned, had to interrupt"

    try:
        res = completion.choices[0].message.content
    except TypeError:
        res = "Error: No response from AI."
    print(f"=========(response from {model_id})=============")
    print(res)
    print("================================================")
    return res





def get_secret(key):

    return os.getenv(key)


class OpenRouterClient:

        
    free_models = [
        "moonshotai/moonlight-16b-a3b-instruct:free",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "google/gemini-2.0-pro-exp-02-05:free",
        "qwen/qwen-vl-plus:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "google/gemini-2.0-flash-thinking-exp:free",
        "deepseek/deepseek-r1:free",
        "sophosympatheia/rogue-rose-103b-v0.2:free",
        "deepseek/deepseek-chat:free",
        "google/gemini-2.0-flash-thinking-exp-1219:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-exp-1206:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "meta-llama/llama-3.2-1b-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-nemo:free",
        "qwen/qwen-2-7b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "meta-llama/llama-3-8b-instruct:free",
        "openchat/openchat-7b:free",
        "undi95/toppy-m-7b:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "gryphe/mythomax-l2-13b:free",
    ]

    paid_models = [
        "liquid/lfm-7b",
        "liquid/lfm-3b",
        "meta-llama/llama-3.1-8b-instruct"
    ]

    model_results = Counter()


    def cache_files_for_gemini(self, files, display_name, system_prompt):
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        contents = []
        for f in files:
            # Upload the video using the Files API
            upload_file = self.gemini_client.files.upload(file=f)

            # Wait for the file to finish processing
            while upload_file.state.name == 'PROCESSING':
                print('Waiting for file to be processed.')
                time.sleep(1)

            contents.append(client.files.get(name=video_file.name))

        print(f'File processing complete: {video_file.uri}')

        # You must use an explicit version suffix. "-flash-001", not just "-flash".
        model='models/gemini-1.5-flash-001'

        # Create a cache with a 5 minute TTL
        self.gemini_cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=display_name, # used to identify the cache
                system_instruction=system_prompt,
                contents=contents,
                ttl="300s",
            )
        )



    async def send_cached_gemini_request(
        self,
        system_prompt,
        content,
        cache_name,
        model="gemini-2.0-flash-exp"):


        response = client.models.generate_content(
            model=model,
            contents=content,
            config=types.GenerateContentConfig(
                cached_content=self.gemini_cache.name)
        )



    async def send_gemini_request(
        self, 
        system_prompt, 
        content, 
        temperature=0.0, 
        model="gemini-2.0-flash-exp"):

        response = await genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
            model="gemini-2.0-flash-exp", contents=[system_prompt, content],
        )
        # print(response.text)
        return response.text


    async def send_openrouter_request(
        self,
        system_prompt, 
        content, 
        temperature=0.0, 
        model="meta-llama/llama-3.1-8b-instruct",
        max_retries=10):

        if max_retries < 0:
            raise Exception("Max retries reached")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


        # model = choice(self.allowed_models)
        # await aioconsole.aprint(f"Using model: {model}")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            model=model,
            messages=[
                {
                "role": "system",
                "content": system_prompt,
                },
                {
                "role": "user",
                "content": content,
                },
            ],
            temperature=temperature,
        )
        # print(system_prompt)
        # print(content)
        try:
            # await aioconsole.aprint(f"completion: {completion}")
            # await aioconsole.aprint(f"completion.choices: {completion.choices}")
            # await aioconsole.aprint(f"{dir(completion)}")
            res = completion.choices[0].message.content
            self.model_results[model] += 1
        except TypeError:
            res = "Error: No response from AI."
            self.model_results[model] -= 1
            return await self.send_openrouter_request(
                system_prompt, content, temperature, max_retries=max_retries - 1)
        # print(res)

        return res



def main():
    return send_openrouter_request("You are a helpful assistant.", "What is the meaning of life?")

if __name__ == "__main__":
    main()