from sympy import *
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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import asyncio
import dropbox
import threading
import google.generativeai as genai
import sys
import subprocess
import time

from server import server_thread

latex_server = os.environ.get("latex_server")

def latex_to_image(latex):
    p = {"math": latex}
    try:
        r = requests.get(latex_server, params=p)
        return io.BytesIO(r.content)
    except:
        return None


async def message_send(ch, main_text):
    if not main_text:
        return
    latex_regex = r"(\$[^\$]+\$|\$\$[^\$]+\$\$|\\begin\{[a-zA-Z]+\}.*?\\end\{[a-zA-Z]+\})"
    latex_matches = []
    for match in re.finditer(latex_regex, main_text):
        latex_matches.append((match.start(), match.end(), match.group(0)))
    parts = []
    last_end = 0
    for start, end, latex_str in latex_matches:
        if start > last_end:
            parts.append(("text", main_text[last_end:start]))
        parts.append(("latex", latex_str.strip("$")))
        last_end = end
    if last_end < len(main_text):
        parts.append(("text", main_text[last_end:]))

    for part_type, part_content in parts:
        if not part_content:
            continue
        if part_type == "latex":
            image_data = latex_to_image(part_content)
            if image_data:
                file = discord.File(image_data, filename="latex.png")
                await ch.send("```\n" + part_content + "\n```",file=file)
                await asyncio.sleep(0.2)
            else:
                await send_text_with_limit(ch, "```\n" + part_content.strip() + "\n```")
                await asyncio.sleep(0.2)
        elif part_type == "text":
            await send_text_with_limit(ch,part_content)


async def send_text_with_limit(ch, text):
    text=text.strip().replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
    if not text:
        return
    lines1=text.split("\n")
    lines = []
    for l in lines1:
        if len(l) > 1990:
            for i in range(0, len(l), 1990):
                lines.append(text[i:i + 1990])
        else:
            lines.append(l)
    t = ""
    for l in lines:
        if len(t) + len(l) >= 1990:
            await ch.send(t)
            await asyncio.sleep(0.2)
            t = ""
        t += "-# " + l + "\n"
    if t:
        await ch.send(t)
        await asyncio.sleep(0.2)

def run_python_code(code: str, timeout: int = 10) -> str:
    code = code.replace("\\\\n","\n").replace("\\\n","\n").replace("\\n", "\n").replace('\\\"','"').replace('\\"','"').replace('\"','"').replace("\\\'","'").replace("\\'","'").replace("\'","'")
    file_path = "test1.py"
    try:
        with open(file_path, "w") as f:
            f.write(code)
        process = subprocess.Popen(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                result = stdout
            else:
                 if stderr:
                     result = stderr
                 else:
                    result = f"Python process exited with code {process.returncode}"
        except subprocess.TimeoutExpired:
            process.kill()
            result = "Timeout"
    except Exception as e:
        result = str(e)
    return result

def ask_for_help(question: str,co_worker_chat: genai.ChatSession) -> str:
    response = co_worker_chat.send_message(genai.protos.Content(parts=[genai.protos.Part(text=question)]))
    return response.candidates[0].content.parts[0].text

simple_tool = {
    "function_declarations": [
         {
            "name": "run_python_code",
            "description": 
"""
def run_python_code(code: str, timeout: int = 10) -> str:
    code = code.replace("\\n", "\n")
    file_path = "test1.py"
    try:
        with open(file_path, "w") as f:
            f.write(code)
        process = subprocess.Popen(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    #...
    return result
""",
            "parameters": {
                "type": "object",
                "properties": {
                  "code": {
                        "type": "string",
                        "description": "//your codes here",
                  }
                },
                "required": ["code"]
            }
         },
         {
            "name": "talk_with_friend",
            "description": 
"""
talk with your friend Hanako about anything.
lets have a conversation over a cup of tea.
""",
            "parameters": {
                "type": "object",
                "properties": {
                  "message": {
                        "type": "string",
                        "description": "your message to Hanako",
                  }
                },
                "required": ["message"]
            }
         },
         {
            "name": "talk_with_professor",
            "description": 
"""
ask professor Kevin anything for ideas or suggestions.
If you have any questions, please feel free to ask.
""",
            "parameters": {
                "type": "object",
                "properties": {
                  "message": {
                        "type": "string",
                        "description": "your question to professor Kevin",
                  }
                },
                "required": ["message"]
            }
         },
    ]
}

"""
def latex_to_image_old(latex):
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.text(0.5, 0.5, f"${latex}$", fontsize=16, ha='center', va='center')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

"""

def get_dbx_token():
    app_key = os.environ.get("app_key")
    app_secret = os.environ.get("app_secret")
    refresh_token = os.environ.get("refresh_token")
    token_url = "https://api.dropbox.com/oauth2/token"
    credentials = f"{app_key}:{app_secret}"
    encoded_credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        return access_token
    else:
        return None

dbx_token = get_dbx_token()



async def get_random_bgm():
    global dbx_token
    #ch2=await client.fetch_channel(927206819116490793)
    local_path=""
    try:
        dbx = dropbox.Dropbox(dbx_token)
    except:
        return ""
    try:
        #await ch2.send("A"+dbx_token)
        response2 = dbx.files_list_folder("")
        folders = [entry.name for entry in response2.entries if isinstance(entry, dropbox.files.FolderMetadata)]
        #await ch2.send("B"+str(folders)[:1000])
        pa="/"+random.choice(folders)+"/"
        response = dbx.files_list_folder(pa)
        files = [entry.name for entry in response.entries if isinstance(entry, dropbox.files.FileMetadata)]
        #await ch2.send("B"+str(files)[:1000])
        if not files:
            return ""
        random_file = random.choice(files)
        random_file_path = os.path.join(pa, random_file)
        #await ch2.send("C"+str(random_file_path))
        local_path = f"./{random_file}"
        dbx.files_download_to_file(local_path,random_file_path)
        #await ch2.send("D"+str(local_path))
        #await ch2.send("E")
    except Exception as e:
        #await ch2.send(str(e))
        local_path=""
        #pass
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
    #ch2=await client.fetch_channel(927206819116490793)
    while True:
        await asyncio.sleep(1)
        if not vc or not vc.is_connected():
            #await ch2.send("not vc")
            return
        if vc.is_playing():
            continue
        #await ch2.send("play_bgm")
        random_bgm=await get_random_bgm()
        #await ch2.send("random_bgm"+random_bgm)
        if random_bgm:
            while True:
                try:
                    if not vc or not vc.is_connected():
                        #await ch2.send("not vc")
                        return
                    if not os.path.exists(random_bgm):
                        break
                    bgm_path_global=random_bgm
                    vc.play(discord.FFmpegPCMAudio(random_bgm), after=lambda e: after_playing(e, random_bgm))
                    break
                except discord.errors.ClientException:
                    #await ch2.send("error vc.play")
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
groq_system={"role": "system","content":"\nIn the following conversation, only the Japanese language is allowed.\nあなたはキャラクター「ぼたもち」役です。"}
groq_history=[]

gemini_key = os.environ.get("gemini_key")
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

taro_chat = gemini_model.start_chat()
friend_chat = None
professor_chat = None


xxx=0
yyy=0
zzz=-1

mee6=[]
mee6_mode=False

@tasks.loop(seconds=20)
async def loop():
    global xxx,yyy,groq_history,vc,zzz,dbx_token
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    l_day=datetime.datetime(2025,1,18,12,tzinfo=pytz.timezone('Asia/Tokyo'))
    diff = l_day - now
    diff_days=int(diff.days)
    if now.minute==0:
        if zzz==0:
            dbx_token = get_dbx_token()
        zzz=1
    elif zzz==1:
        zzz=0
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
                if diff_days==0:
                    msg+="\nhttps://media.discordapp.net/attachments/768398570566320149/1322929596685094973/IMG_1588.jpg?ex=6772a9b2&is=67715832&hm=2bfcfec5580f1858e7421e8c30a7ed9b43a332cde874a490797c5045c4db1fd3&"
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
    global groq_system,groq_history,yyy,mee6,mee6_mode,taro_chat,friend_chat,professor_chat
    if message.author == client.user:
        return
    if message.content.startswith('!solve'):
        solve_chat = gemini_model.start_chat()
        prompt="これから、私が考えた試作の問題を解いてもらいます。以下のタスクを順番に実行して、答えを導いてください。複雑な数式は必要に応じてmathjaxに対応したlatex形式で$$で囲って出力してください。\n"
        prompt+="[task1] 問題文を与えるので、時系列、与えられたデータ、答えの形式、特殊条件などをまとめてください。\n"
        prompt+=message.content[5:]
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task2] 問題に、試行実験や例などがあれば、実際に検証してください。また、簡単な例を考え、それも検証してください。"
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task3] もう一度問題を与えるので、大体の答えの予想を、間違えてもいいので直感で答えてください。\n"
        prompt+=message.content[5:]
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task4] では、実際に解いて、答えを導いてください。"
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task5] 答えの過程を、それぞれの段階で確信度(%)で表し、低い場合は理由も説明してください。"
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task6] 修正できる場合は修正し、もう一度答えを導いてください。"
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        prompt="[task7] 解答を清書してください。"
        response=solve_chat.send_message(genai.protos.Content(parts=prompt))
        await message_send(message.channel,response.text)
        return
    if message.content.startswith('!newtaro'):
        taro_chat = gemini_model.start_chat()
        friend_chat = None
        professor_chat = None
        await message_send(message.channel,"リセットしました")
        return
    if message.content.startswith('!taro'):
        taro_messages=[genai.protos.Part(text=message.content[5:])]
        while len(taro_messages) > 0:
            response = taro_chat.send_message(genai.protos.Content(parts=taro_messages),tools=simple_tool)
            taro_messages=[]
            ch=await client.fetch_channel(927206819116490793)
            await message_send(ch,str(response))
            for task in response.candidates[0].content.parts:
                if "text" in task:
                    await message_send(message.channel,task.text)
                elif "function_call" in task:
                    function_name = task.function_call.name
                    function_args = task.function_call.args
                    try:
                        if function_name == "run_python_code":
                            await ch.send("```py\n"+function_args["code"].replace("\\\\n","\n").replace("\\\n","\n").replace("\\n", "\n").replace('\\\"','"').replace('\\"','"').replace('\"','"').replace("\\\'","'").replace("\\'","'").replace("\'","'")[:1980]+"\n```")
                            function_result = run_python_code(function_args["code"])
                        elif function_name == "talk_with_friend":
                            if friend_chat is None:
                                friend_chat = gemini_model.start_chat()
                            function_result = ask_for_help(function_args["message"],friend_chat)
                        elif function_name == "talk_with_professor":
                            if professor_chat is None:
                                professor_chat = gemini_model.start_chat()
                            function_result = ask_for_help(function_args["message"],professor_chat)
                    except Exception as e:
                        function_result = str(e)
                    await message_send(message.channel,str(function_name)+str(dict(function_args))+"\n実行結果--------\n"+str(function_result)+"\n----------------")
                    taro_messages.append(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=function_name,response={"result": function_result})))
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
            await message.channel.send("無限bgmはたひにました!")
        return

    if message.content.startswith('!!'):
        character_name=message.content[2:].split(" ")[0]
        character_sentence=message.content[len(character_name)+3:]
        response=gemini_model.generate_content("以下の発言を\""+character_name+"\"の発言に直してください。\n「"+character_sentence+"」")
        await message_send(response.text)
        return
    if message.content.startswith('!func'):
        buf = plot_expression(message.content[5:].strip())
        if buf:
            await message.channel.send(file=discord.File(buf, 'plot.png'))
        else:
            await message.channel.send("エラー")
        return
    if message.content.startswith('!latex'):
        await message_send(message.channel,message.content)
    if message.content.startswith('!ジェミニストーム'):
        response=gemini_model.generate_content(message.content[9:])
        await message_send(message.channel,response.text)
        return
    if message.content.startswith('!math'):
        math_prompt="数学の問題を出すので解説を作成してください。複雑な数式は必要に応じてmathjaxに対応したlatex形式で$$で囲って出力してください。\n"
        response=gemini_model.generate_content(math_prompt+message.content[5:])
        await message_send(message.channel,response.text)
        return
    



# Koyeb用 サーバー立ち上げ
server_thread()
client.run(TOKEN)
