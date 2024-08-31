# MTek SavAIr
IMPORTANT: This is a weekend project with strong coffee Js. Presently, 'just for friends' development stage, WIP.
We're still trying to figure things out. 😉

---

## Hers's the Plan: (Hold the Beer)

Provide AI tooling for E-MTECH_CC students to learn with AI. Abstracting AI tech from them to focus on the course itself.
Aimed at being student's AI assistant, specilizing in course material of e-MTECH-CC IIT Patna.
Provide a framework to play with & also personlize learning as per their temperment and AI powered.
Integrate/Encapsulate Latest AI tech for making learning easier for E-MTECH students.
Integrate personlity to AI to tailor response as per each students liking, level etc. Hyper personlized learning.
Personalized learning: It'll tailor your learning experience. No one will ever know you don't understand anything!
Hyper-personalized learning: It's like reading your mind... but we haven't quite figured that part out yet.

Note: The AI knows more than you. A Super-teacher who will never tire of your questions and tailor response as you would like to learn.

---

## 🏃How to Run This Magical Thing: 🏃

For MTECH students:

- Clone this repo. (You know how to do that, right?)
- Install the requirements.txt (If you don't know what that is, maybe opt E-MBA).
- streamlit run emtech_cc.py. (Duh! Make sure you're in the right directory!)
- Visit the web URL. (Hopefully, it'll work... fingers crossed!).

---

## Features & Usecases for E-MTECH-CC Course:

1). Input slide page contents and ask questions, clarifications, examples etc
2). Input course video for speech to text, get summary, 'what the professor said', 'student questions' etc
Note: 'Sir, please stop video recording', not required as the AI model recognises the main speakers and only transcribes their part.
This is speech diarization.
3). Python code interpetor is present and hence AI can code along with you and explain programming, 'n' number of times.
4). There are inputs which are present to abstract prompting and provide right answers.
5). Multilingual : Convert speech, text, slides to any language you are most comfortable with.
6). Supports Text, Video, Image from PDF & PPT files.

NOTE: To Err is Human & AI...

---

## 🚧 Implementation Details & Design considerations:

1). Avoid Using standard RAG libraries like langchain, llamaindex, dify, haystack etc. 
Implementation is easier, community support & sh-t-like-that. 
Heavy libs, lesser customizations and whats the fun in using them.

2). Lets build our own RAG and make mistakes around it.

3). PPTs, PDFs, videos are unstructured data and difficult to preprocess for data. 
Different vendors, formats, codecs make it difficult to parse all. llamaparse is good option, after we fail at it.

4). Multimodal AI models are great at extracting data from unstructured data. Imagine giving eyes to your programs.

5). More powerful AI models will give better results, Get your own expensive APIs. Cheapest-APIs-first.

6). Python-Only approach for full stack streamlit app. 🤘 We're keeping it simple... for now.

---

## 🚀 RoadMap

1). AssemblyAI integration - Speech to text analysis
2). Support Gemini & OpenAI AI models.
3). Experiment with different RAG framework & Benchmark accuracy.
4). Learn with speech & Visualizations for course material.
5). Chat with your EMTECH-CC-IITPatna syllabus and its course materials.
6). Provide feedback to teachers of how students are learning.
7). Chat with classroom video recordings and ask questions, get details, links etc.
8). Embed microsft chats to get useful messages and store in db.
9). Learn C & python programming.
10). Fine-Tune AI models with prompting.
11). Better react-based UI.

---

Irony: We started this project, as finding time is Not as simple as 'Import time'.
Please contribute, Dont ask questions, features, Criticise. In Active Developement.
Remember - Its a Free tool & NOT included in your course fee.
