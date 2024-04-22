import discord
import dotenv
import os
import datetime
import pytz

from server import server_thread

dotenv.load_dotenv()

TOKEN = os.environ.get("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

xxx=0

@tasks.loop(seconds=50)
async def loop():
    global xxx
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    if now.hour == 1 and now.minute == 1:
        if xxx==0:
            ch=await client.fetch_channel(927206819116490793)
            await ch.send('<@&928999554290970634>')
        xxx=1
    else:
        xxx=0

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        ch=await client.fetch_channel(927206819116490793)
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        await ch.send('Hello!'+str(now))

# Koyeb用 サーバー立ち上げ
server_thread()
client.run(TOKEN)
