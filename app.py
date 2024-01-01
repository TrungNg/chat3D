import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import openai
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils import Shap_e_TextTo3D
from langchain.chat_models import ChatOpenAI
#from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

memory = ConversationBufferWindowMemory(k=100)

system_template = SystemMessagePromptTemplate.from_template("You are an assistant in 3D modelling")

CONVER_MODEL = "gpt-3.5-turbo"
llm = ChatOpenAI(
    model_name=CONVER_MODEL,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], '_.jpg')
render_file_path = os.path.join('static', 'sample.gif')
model = Shap_e_TextTo3D()
latents = None
all_session = ""
IDLE, GENERATING_FROM_TEXT, GENERATING_FROM_IMAGE = range(3)

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        global img_file_path
        global latents
        global all_session
        if request.form.get("action", None) == "Material?":
            user_template = HumanMessagePromptTemplate.from_template("{prompt}")
            chat_template = ChatPromptTemplate.from_messages([system_template, user_template])
            response = LLMChain(llm=llm, prompt=chat_template).run({"prompt": "What is it usually made of?"})
            return redirect(url_for("index", result=response)) #['choices'][0]['message']['content']
        
        elif  request.form.get("action", None) == "Scale?":
            user_template = HumanMessagePromptTemplate.from_template("{prompt}")
            chat_template = ChatPromptTemplate.from_messages([system_template, user_template])
            response = LLMChain(llm=llm, prompt=chat_template).run({"prompt":"What is its typical length?"})
            return redirect(url_for("index", result=response)) #['choices'][0]['message']['content']
        
        elif request.form.get("action", None) == "Generate 3D":
            user_template = HumanMessagePromptTemplate.from_template("{prompt}")
            chat_template = ChatPromptTemplate.from_messages([system_template, user_template])
            prompt_3d = LLMChain(llm=llm, prompt=chat_template).run({"prompt": generate_3d_prompt(all_session)})
            print(prompt_3d)
            latents = model.gen_from_text(prompt_3d)
            return redirect(url_for("index", result=prompt_3d + "\n3D model generated!", model_stage=IDLE))
        
        elif request.form.get("action", None) == "Save Meshes":
            model.saveMeshes(latents)
            return redirect(url_for("index", result="Saved 3D model as meshes"))
        
        elif request.form.get("action", None) == "New Session":
            img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], '_.jpg')
            if os.path.exists(render_file_path):
                os.remove(render_file_path)
            user_template = HumanMessagePromptTemplate.from_template("{prompt}")
            chat_template = ChatPromptTemplate.from_messages([system_template, user_template])
            response = LLMChain(llm=llm, prompt=chat_template).run({"prompt": 'Forget all the above chat and start a fresh session.'})

            conversation.memory.clear()
            return redirect(url_for("index", result=response))
        
        elif request.form.get("img_upload") == "Upload":
            # Upload file flask
            uploaded_img = request.files['uploaded-file']
            # Extracting uploaded data file name
            img_filename = secure_filename(uploaded_img.filename)
            img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            # Upload file to database (defined uploaded folder in static path)
            uploaded_img.save(img_file_path)
            return redirect(url_for("index", result="Upload done!"))
        
        elif request.form.get("imaction", None) == "Generate from image":
            if os.path.exists(img_file_path):
                print("generating from image")
                latents = model.gen_from_image(img_file_path)
            return redirect(url_for("index", result="3D model generated from input image!", model_stage=IDLE))

        else:
            prompt = request.form["prompt"]
            conversation.run(prompt)
            return redirect(url_for("index", result=response['choices'][0]['message']['content']))
    
    ifp = None
    rfp = None
    if os.path.exists(img_file_path):
        ifp = img_file_path
    if os.path.exists(render_file_path):
        rfp = render_file_path
    return render_template("index.html", result=request.args.get("result"), upload_image=ifp, render_3d=rfp)


def generate_3d_prompt(session):
    return """{}\nExtract the most relevant term from your latest response to my most recent prompt and answer with only the term. You can also add any description to your answer if it helps the 3D modelling, such as dimensions and shape ratios, but only the details in your answer""".format(session)

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
