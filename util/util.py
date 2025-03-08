import os
from openai import OpenAI
from dotenv import load_dotenv

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
