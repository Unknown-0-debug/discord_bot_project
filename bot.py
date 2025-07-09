import os
import discord
from discord.ext import commands
import nest_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import requests
import json
import torch
from groq import Groq
import asyncio

nest_asyncio.apply()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

chatbot = ChatBot(
    'SAO Bot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///:memory:',
    read_only=True
)
trainer = ListTrainer(chatbot)
trainer.train([
    "What is SAOVS?",
    "SAOVS is a mobile application made by Bandai Namco based on the Sword Art Online anime series. It is an official game, featuring characters from the anime in multiplayer PvP and PvE battles with real-time action.",
    "How do I start playing SAOVS?",
    "After downloading the game and its data, you start by completing the tutorial. You get a free scout with a retry option — you can keep scouting until you get the character you want, but only the final scout counts.",
    "What platforms is SAOVS available on?",
    "SAOVS can be downloaded from the Google Play Store or the Apple App Store, and it’s available for mobile devices.",
    "How do I get new characters?",
    "You get new characters via scouting, which lets you pull characters and ability cards by spending Variant Crystals (VC).",
    "What is scouting?",
    "You get new characters via scouting, which lets you pull characters and ability cards by spending Variant Crystals (VC).",
    "How does the scout system work?",
    "A normal scout costs 150 VC per pull, and a 10x scout costs 1500 VC and guarantees at least an SR or higher rarity item (character or ability). Step-up scouts offer only 10x pulls, costing 1000 VC on step 1, 1200 VC on step 2, and 1500 VC on steps 3 to 5.",
    "How to guarantee a character from scouting?",
    "To guarantee a character, you need to ‘pity’ by exchanging scout points earned—1 point per scout. You need 200 points to guarantee a character, which means spending about 30,000 VC on normal scouts or 26,800 VC on step-up scouts to guarantee a character.",
    "What are ability cards?",
    "Ability cards are available in scouts and require the same pity system of the characters to guarantee them."
])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

chat_histories = {}
user_modes = {}

MAX_DISCORD_MSG_LENGTH = 2000

def split_message(text, max_length=MAX_DISCORD_MSG_LENGTH):
    words = text.split(' ')
    chunks, current_chunk = [], ""
    for word in words:
        if len(current_chunk) + len(word) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk += (" " if current_chunk else "") + word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

async def send_long_message(channel, message):
    for chunk in split_message(message):
        await channel.send(chunk)

def openrouter_generate(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"OpenRouter API error {response.status_code}: {response.text}")
        return "Sorry, I couldn't process that right now."

def groq_generate(prompt: str) -> str:
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return "Sorry, Groq model is currently unavailable."

@bot.command()
async def switch(ctx):
    uid = ctx.author.id
    current = user_modes.get(uid, "dialo")
    next_mode = {"dialo": "openai", "openai": "groq", "groq": "dialo"}
    new_mode = next_mode.get(current, "dialo")
    user_modes[uid] = new_mode
    await ctx.send(f"✅ Switched to **{new_mode.upper()}** mode for you.")

@bot.command()
async def reset(ctx):
    uid = ctx.author.id
    chat_histories.pop(uid, None)
    await ctx.send("✅ Your chat history has been reset.")

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}!")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    if bot.user.mentioned_in(message):
        user_id = message.author.id
        prompt = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()

        if not prompt:
            await message.channel.send("Yes? How can I help you?")
            return

        response = chatbot.get_response(prompt)
        if float(response.confidence) >= 0.80:
            await send_long_message(message.channel, str(response))
            return

        mode = user_modes.get(user_id, "dialo")

        if mode == "dialo":
            new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
            if user_id in chat_histories:
                bot_input_ids = torch.cat([chat_histories[user_id], new_input_ids], dim=-1)
                if bot_input_ids.shape[-1] > 1024:
                    bot_input_ids = bot_input_ids[:, -1024:]
            else:
                bot_input_ids = new_input_ids

            attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
            chat_histories[user_id] = model.generate(
                bot_input_ids,
                attention_mask=attention_mask,
                max_length=1000,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_p=0.9,
                top_k=50,
            )
            reply = tokenizer.decode(chat_histories[user_id][0, bot_input_ids.shape[-1]:], skip_special_tokens=True)
            await send_long_message(message.channel, reply)

        elif mode == "openai":
            reply = openrouter_generate(prompt)
            await send_long_message(message.channel, reply)

        else:
            reply = groq_generate(prompt)
            await send_long_message(message.channel, reply)

bot.run(os.getenv("DISCORD_TOKEN"))
