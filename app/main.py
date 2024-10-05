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
import re
import networkx as nx
from graphillion import GraphSet
import graphillion.tutorial as tl
import base64
import io
import sympy as sp
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import dropbox
import threading

from server import server_thread

dbx_token = os.environ.get("dbx_token")
print(dbx_token)

async def get_random_bgm():
    global dbx_token
    ch2=await client.fetch_channel(927206819116490793)
    local_path=""
    pa="/kirby_mix"
    try:
        dbx = dropbox.Dropbox(dbx_token)
    except:
        return ""
    try:
        await ch2.send("A"+dbx_token)
        response = dbx.files_list_folder(pa)
        files = [entry.name for entry in response.entries if isinstance(entry, dropbox.files.FileMetadata)]
        await ch2.send("B"+str(files)[:1000])
        if not files:
            return ""
        random_file = random.choice(files)
        random_file_path = os.path.join(pa, random_file)
        await ch2.send("C"+str(random_file_path))
        local_path = f"./{random_file}"
        dbx.files_download_to_file(local_path,random_file_path)
        await ch2.send("D"+str(local_path))
        await ch2.send("E")
    except Exception as e:
        await ch2.send(str(e))
        local_path=""
        pass
    return local_path

vc=None
bgm_path_global=""

def after_playing(error,bgm_path):
    global bgm_path_global
    if bgm_path:
        if os.path.exists(bgm_path):
            os.remove(bgm_path)
    elif bgm_path_global:
        if os.path.exists(bgm_path_global):
            os.remove(bgm_path_global)

async def play_bgm():
    global vc,bgm_path_global
    ch2=await client.fetch_channel(927206819116490793)
    while True:
        await asyncio.sleep(1)
        if not vc or not vc.is_connected():
            await ch2.send("not vc")
            return
        if vc.is_playing():
            continue
        await ch2.send("play_bgm")
        random_bgm=await get_random_bgm()
        await ch2.send("random_bgm"+random_bgm)
        if random_bgm:
            while True:
                try:
                    if not vc or not vc.is_connected():
                        await ch2.send("not vc")
                        return
                    if not os.path.exists(random_bgm):
                        break
                    bgm_path_global=random_bgm
                    vc.play(discord.FFmpegPCMAudio(random_bgm), after=lambda e: after_playing(e, random_bgm))
                    break
                except discord.errors.ClientException:
                    await ch2.send("error vc.play")
                    await asyncio.sleep(1)
        else:
            await asyncio.sleep(5)
        

def plot_expression(expression_str):
    try:
        x = sp.Symbol('x')
        expression = sp.sympify(expression_str)
        f = sp.lambdify(x, expression, 'numpy')
        x_vals = np.linspace(-10, 10, 1001)
        y_vals = f(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)
        mask = np.isfinite(y_vals) & (np.abs(np.diff(y_vals, prepend=np.nan)) < 20) & (np.abs(np.diff(y_vals, append=np.nan)) < 20)
        x_segments = np.split(x_vals, np.where(~mask)[0])
        y_segments = np.split(y_vals, np.where(~mask)[0])
        plt.figure(figsize=(10, 8))
        for x_seg, y_seg in zip(x_segments, y_segments):
            if len(x_seg) > 1:
                plt.plot(x_seg, y_seg, zorder=2)
            elif len(x_seg) == 1:
                plt.scatter(x_seg, y_seg, zorder=2)
        plt.xlabel('x')
        plt.ylabel(f'f(x) = {expression_str}')
        plt.title(f'Graph of {expression_str}')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.axhline(0, color='black', linewidth=2, zorder=1)
        plt.axvline(0, color='black', linewidth=2, zorder=1)
        plt.xticks(np.arange(-10, 11, 1))
        plt.yticks(np.arange(-10, 11, 1))
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except (sp.SympifyError, ValueError):
        return None

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
    global xxx,yyy,groq_history,vc
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
                msg='あと'+str(diff_days)+'日'
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
    global vc
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
    if message.content.startswith('!bgm'):
        if message.author.voice is None:
            await message.channel.send("ボイスチャンネルに参加してね")
            return
        vc=await message.author.voice.channel.connect()
        await message.channel.send(str(message.author.voice.channel) + "に接続したので、!killでたひにます")
        asyncio.create_task(play_bgm())
        return
    if message.content.startswith('!kill'):
        if vc:
            try:
                if vc.is_playing():
                    vc.stop()
                await vc.disconnect()
            except:
                pass
            after_playing(None,None)
            vc = None
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
        groq_history=[]
        groq_system={"role": "system","content": "In the following conversation, only the Japanese language is allowed."+message.content[9:]}
        await message.channel.send(groq_system["content"])
        return
    if message.content.startswith('!おねえさん'):
        parts = re.split(r'\D+',message.content)
        nums=[]
        for ppp in parts:
            if ppp:
                try:
                    nums.append(int(ppp))
                except:
                    pass
        if len(nums)!=2:
            await message.channel.send("エラー "+str(nums))
            return
        if nums[0]>6 or nums[1]>6 or nums[0]<1 or nums[1]<1:
            await message.channel.send("1以上6以下 "+str(nums))
            return
        universe = tl.grid(nums[0],nums[1])
        GraphSet.set_universe(universe)
        start = 1
        end = (nums[0]+1)*(nums[1]+1)
        paths = GraphSet.paths(start, end)
        total_paths = paths.len()
        await message.channel.send(f"{nums[0]}x{nums[1]}格子グラフの総経路数: {total_paths}")
    if message.content.startswith('!!'):
        character_name=message.content[2:].split(" ")[0]
        character_sentence=message.content[len(character_name)+3:]
        next_messages=[{"role": "system","content": "In the following conversation, only the Japanese language is allowed."}]
        next_chat={"role": "user", "content": "以下の発言を\""+character_name+"\"の発言に直してください。\n「"+character_sentence+"」"}
        next_messages.append(next_chat)
        response = groq_client.chat.completions.create(model="llama-3.1-70b-versatile",
                                            messages=next_messages,
                                            max_tokens=720,
                                            temperature=1.05)
        await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
        return
    """
    if message.content.startswith('!ぼたもち画像'):
        img_64=""
        try:
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type.startswith("image"):
                        img_bytes = await attachment.read()
                        img_64=base64.b64encode(img_bytes).decode("utf-8")
                        break
        except:
            img_64=""
        if img_64=="":
            await message.channel.send("画像を送ってください")
            return
        next_messages=[{"role": "system","content": "In the following conversation, only the Japanese language is allowed."}]
        next_chat={"role": "user", "content":message.content[5:],"image":img_64}
        next_messages.append(next_chat)
        response = groq_client.chat.completions.create(model="llava-v1.5-7b-4096-preview",
                                            messages=next_messages,
                                            max_tokens=720,
                                            temperature=1.05)
        await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
        return
    """
    if message.content.startswith('!func'):
        buf = plot_expression(message.content[5:].strip())
        if buf:
            await message.channel.send(file=discord.File(buf, 'plot.png'))
        else:
            await message.channel.send("エラー")
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
            if message.content.startswith('!ぼたもち'):
                await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
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
