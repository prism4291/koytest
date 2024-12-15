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

async def message_send(ch,main_text):
    main_text=str(main_text).strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
    if len(main_text)>0:
        lines=main_text.split("\n")
        t=""
        for l in lines:
            if len(t)+len(l)>=1990:
                await ch.send(t)
                t=""
            t+="-# "+l+"\n"
        await ch.send(t)


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


def latex_to_image(latex):
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.text(0.5, 0.5, f"${latex}$", fontsize=16, ha='center', va='center')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

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

META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()


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
    if message.content=="!gacha":
        mee6_mode=True
        return
    if mee6_mode and message.author.id == 159985870458322944:
        mee6.append(message.content)
        if len(mee6)==5:
            mee6_str="\n".join(mee6)
            next_messages=[{"role": "system","content": "In the following conversation, only the Japanese language is allowed.\nあなたは日本の昔話の小説家です。日本語で答えてください。"},{"role": "user", "content":"あらすじを考えました。物語を日本語で書いてください。"+mee6_str}]
            response = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
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
            await message_send(ch,response)
            for task in response.candidates[0].content.parts:
                if "text" in task:
                    await message_send(message.channel,task.text)
                elif "function_call" in task:
                    function_name = task.function_call.name
                    function_args = task.function_call.args
                    #print(function_name, dict(function_args))
                    try:
                        if function_name == "run_python_code":
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
                    await message_send(message.channel,str(function_name)+(str(dict(function_args)).replace("\\\\n","\n").replace("\\\n","\n").replace("\\n","\n"))+"\n実行結果--------\n"+str(function_result)+"\n----------------")
                    taro_messages.append(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=function_name,response={"result": function_result})))
            #if len(taro_messages) == 0:
            #    user_text=input(">").strip()
            #    if user_text!="":
            #        taro_messages.append(genai.protos.Part(text=user_text))
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
        response = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
                                            messages=next_messages,
                                            max_tokens=720,
                                            temperature=1.05)
        await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
        return
    if message.content.startswith('!func'):
        buf = plot_expression(message.content[5:].strip())
        if buf:
            await message.channel.send(file=discord.File(buf, 'plot.png'))
        else:
            await message.channel.send("エラー")
        return
    if message.content.startswith('!tex'):
        response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": 'You are a helpful assistant that converts mathematical expressions to TeX format. Please return the TeX output in a raw string format (e.g., r"..." use double quotes).'
                        },
                        {
                            "role": "user",
                            "content": 'Convert the following mathematical expression to TeX: r"'+message.content[6:].strip()+'". Please provide the output as a raw string.',
                        }
                    ],
                    temperature=0.85,
                    max_tokens=1024,
                )
        await message.channel.send(response.choices[0].message.content)
        match_list = re.findall(r'r"([^"]*)"', str(response.choices[0].message.content))
        if len(match_list)==0:
            match_list = re.findall(r"r'([^']*)'", str(response.choices[0].message.content))
        if len(match_list)>0:
            extracted_latex = match_list[-1]
            buf=None
            try:
                buf=latex_to_image(extracted_latex.strip().strip("$"))
            except:
                await message.channel.send("エラー1 cannot create")
                return
            if buf:
                await message.channel.send(file=discord.File(buf, 'tex.png'))
                return
            else:
                await message.channel.send("エラー2 empty")
                return
        else:
            await message.channel.send("エラー3 cannot find tex もう一度試してみて")
        return
    if message.content.startswith('!ジェミニストーム'):
        response=gemini_model.generate_content(message.content[9:])
        lines=("-# "+response.text.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# ")).split("\n")
        t=""
        for l in lines:
            if len(t)+len(l)>=2000:
                await message.channel.send(t)
                t=""
            t+=l+"\n"
        await message.channel.send(t)
        return
    if message.content.startswith('!math'):
        math_prompt="数学の問題を出すので解説を作成してください。複雑な数式は必要に応じてmatplotlibのmathtext形式で$$で囲ってください。\n"
        response=gemini_model.generate_content(math_prompt+message.content[5:])
        response_text=response.text.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("$$","$")
        latexs=response_text.split("$")
        for i in range(0,len(latexs),2):
            main_text=latexs[i].strip().strip("\n").strip()
            if len(main_text)>0:
                lines=main_text.split("\n")
                t=""
                for l in lines:
                    if len(t)+len(l)>=1990:
                        await message.channel.send(t)
                        t=""
                    t+="-# "+l+"\n"
                await message.channel.send(t)
            if len(latexs)>i+1:
                buf=None
                try:
                    buf=latex_to_image(latexs[i+1].strip("$").strip().strip("\n").strip())
                except:
                    pass
                    #await message.channel.send(latexs[i+1].strip("$"))
                    #return
                if buf:
                    await message.channel.send(file=discord.File(buf, 'tex.png'))
                    #return
                else:
                    await message.channel.send(latexs[i+1].strip("$"))
                    #return
        #else:
        #    await message.channel.send("エラー3 cannot find tex もう一度試してみて")
        return
    if message.content.startswith('!ぼたもち'):# or message.channel.id==1211621332643749918:
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
                
        try:
            if yyy<0:
                yyy+=1
                await message.channel.send("rate limit")
                return
            
            if len(groq_history)>20:
                groq_history=groq_history[-20:]
            if img_64:
                todo_message=""
                next_chat={
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "画像には何が見えますか？"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_64}",
                            },
                        }
                    ]
                }
                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[next_chat],
                    max_tokens=480,
                )
                todo_message+=response.choices[0].message.content
                #await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
                #groq_history.append({"role": "user", "content": "画像には何が見えますか？"})
                #groq_history.append({"role": "assistant","content": response.choices[0].message.content})
                next_chat={
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "画像には何か書かれていますか？"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_64}",
                            },
                        }
                    ]
                }
                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[next_chat],
                    max_tokens=240,
                )
                todo_message+=response.choices[0].message.content
                #await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
                #groq_history.append({"role": "user", "content": "画像には何か書かれていますか？"})
                #groq_history.append({"role": "assistant","content": response.choices[0].message.content})
                #next_messages=[groq_system]
                #next_messages.extend(groq_history)
                next_chat={
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "何を伝えたいのかまとめてください。"+message.content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_64}",
                            },
                        }
                    ]
                }
                #next_messages.append(next_chat)
                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[next_chat],
                    max_tokens=240,
                )
                todo_message+=response.choices[0].message.content
                #next_messages=[groq_system]
                #next_messages.extend(groq_history)
                next_chat={
                    "role": "user",
                    "content": "以下の文章を整形してください。\n(start)\n"+todo_message+"\n(end)"
                }
                #next_messages.append(next_chat)
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[next_chat],
                    temperature=0.5,
                    max_tokens=1024,
                )
                await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
                groq_history.append({"role": "user", "content": "画像を送信しました。"+message.content})
                groq_history.append({"role": "assistant","content": response.choices[0].message.content})
                return
            #if not message.content.startswith('!ぼたもち'):
            #    return
            next_messages=[groq_system]
            next_messages.extend(groq_history)
            next_chat={"role": "user", "content": "「"+str(message.author)+"」さん:「"+message.content+"」"}
            next_messages.append(next_chat)
            response = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
                                            messages=next_messages,
                                            max_tokens=720,
                                            temperature=0.85)
            await message.channel.send("-# "+response.choices[0].message.content.strip().replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n").replace("\n","\n-# "))
            groq_history.append(next_chat)
            groq_history.append({"role": "assistant","content": response.choices[0].message.content})
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
