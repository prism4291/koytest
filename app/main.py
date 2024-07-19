import discord
from discord.ext import tasks
import dotenv
import os
import datetime
import pytz
import random
import requests
import json
from groq import Groq

from server import server_thread

dotenv.load_dotenv()

TOKEN = os.environ.get("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_system={"role": "system","content": "In the following conversation, only the Japanese language is allowed.あなたはキャラクター「ぼたもち」役です。"}
groq_history=[]

xxx=0
yyy=0

mee6=[]
mee6_mode=False

@tasks.loop(seconds=20)
async def loop():
    global xxx,yyy,groq_history
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    l_day=datetime.datetime(2025,1,18,12,tzinfo=pytz.timezone('Asia/Tokyo'))
    diff = l_day - now
    diff_days=int(diff.days)
    if now.hour == 0 and now.minute == 0:
        if xxx==0:
            ch=await client.fetch_channel(768398570566320149)
            if diff_days>=0:
                chg=ch.guild
                chu=client.get_guild(chg.id)
                await chu.me.edit(nick='あと'+str(diff_days)+'日')
        xxx=1
    elif now.hour == 6 and now.minute == 0:
        if xxx==0:
            ch=await client.fetch_channel(768398570566320149)
            if diff_days>=0:
                msg='<@&1231962872360468491>'
                await ch.send(msg)
        xxx=1
    else:
        xxx=0
    
    

@client.event
async def on_ready():
    loop.start()
    ch=await client.fetch_channel(927206819116490793)
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    await ch.send('Hello!'+str(now))

@client.event
async def on_message(message):
    global groq_system,groq_history,yyy,mee6,mee6_mode
    if message.author == client.user:
        return
    if message.content=="!gacha":
        mee6_mode=True
        return
    if mee6_mode and message.author.id == 159985870458322944:
        mee6.append(message.content)
        if len(mee6)==5:
            mee6_str="\n".join(mee6)
            next_messages=[{"role": "system","content": "In the following conversation, only the Japanese language is allowed.\nあなたは日本の昔話の小説家です。日本語で答えてください。"},{"role": "user", "content":"あらすじを考えました。物語を日本語で書いてください。"+mee6_str}]
            response = groq_client.chat.completions.create(model="gemma2-9b-it",
                                            messages=next_messages,
                                            max_tokens=1000,
                                            temperature=1.2)
            message_split=[]
            message_to_split=response.choices[0].message.content.replace("\n\n","\n")
            while len(message_to_split)>1900:
                message_split.append(message_to_split[0:1900])
                message_to_split=message_to_split[1900:]
            message_split.append(message_to_split)
            for ms in message_split:
                await message.channel.send("-# "+ms.strip().replace("\n","\n-# "))
            mee6=[]
            mee6_mode=False
        return
    if message.content.startswith('!ぼたもちストップ'):
        await message.channel.send("stop 5min")
        yyy=0
        return
    if message.content.startswith('!ぼたもちリセット'):
        await message.channel.send("reset "+str(len(groq_history)))
        yyy=0
        groq_history=[]
        groq_system={"role": "system","content": "In the following conversation, only the Japanese language is allowed.あなたはキャラクター「ぼたもち」役です。"}
        return
    if message.content.startswith('!ぼたもちシステム'):
        groq_system={"role": "system","content": "In the following conversation, only the Japanese language is allowed."+message.content[9:]}
        await message.channel.send(groq_system["content"])
        return
    if message.content.startswith('!ぼたもち') or message.channel.id==1211621332643749918:
        try:
            if yyy<0:
                yyy+=1
                await message.channel.send("rate limit")
                return
            next_chat={"role": "user", "content": "「"+str(message.author)+"」さん:「"+message.content+"」"}
            if len(groq_history)>20:
                groq_history=groq_history[-20:]
            next_messages=[groq_system]
            next_messages.extend(groq_history)
            next_messages.append(next_chat)
            response = groq_client.chat.completions.create(model="gemma2-9b-it",
                                            messages=next_messages,
                                            max_tokens=360,
                                            temperature=1.2)
            groq_history.append(next_chat)
            groq_history.append({"role": "assistant","content": response.choices[0].message.content})
            await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
            #ch=await client.fetch_channel(1252576904301510656)
            #await ch.send(response.choices[0].message.content)
            #ch=await client.fetch_channel(1252624652875075697)
            #await ch.send(response.choices[0].message.content)
        except Exception as e:
            ch=await client.fetch_channel(927206819116490793)
            ee=str(e)
            if len(ee)>2000:
                ee=ee[:900]+"\n\n"+ee[-900:]
            await ch.send("onmessage\n"+ee)
            


# Koyeb用 サーバー立ち上げ
server_thread()
client.run(TOKEN)
