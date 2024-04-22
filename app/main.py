import discord
import dotenv
import os

from server import server_thread

dotenv.load_dotenv()

TOKEN = os.environ.get("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        ch=await client.fetch_channel(927206819116490793)
        await ch.send('Hello!')

# Koyeb用 サーバー立ち上げ
server_thread()
client.run(TOKEN)
