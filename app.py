import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import openai
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils import Shap_e_TextTo3D


app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

CONVER_MODEL = "gpt-3.5-turbo"

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
            prompt = "What is it usually made of?"
            response = openai.ChatCompletion.create(
                model=CONVER_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant in 3D modelling."},
                    {"role": "user", "content": prompt}
                    ]
                )
            return redirect(url_for("index", result=response['choices'][0]['message']['content']))
        
        elif  request.form.get("action", None) == "Scale?":
            prompt = "What is its typical length?"
            response = openai.ChatCompletion.create(
                model=CONVER_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant in 3D modelling."},
                    {"role": "user", "content": prompt}
                    ]
                )
            return redirect(url_for("index", result=response['choices'][0]['message']['content']))
        
        elif request.form.get("action", None) == "Generate 3D":
            response = openai.ChatCompletion.create(
                model=CONVER_MODEL,
                messages=[
                    #{"role": "system", "content": "You are an assistant in 3D modelling."},
                    {"role": "user", "content": generate_3d_prompt(all_session)}
                    ]
                )
            prompt_3d = response['choices'][0]['message']['content']
            print(prompt_3d)
            latents = model.gen_from_text(prompt_3d)
            return redirect(url_for("index", result=response['choices'][0]['message']['content'] + "\n3D model generated!", model_stage=IDLE))
            #return redirect(url_for("index", result=response['choices'][0]['message']['content'] + "\nWaiting for creating a model and rendering...", model_stage=GENERATING_FROM_TEXT))
        
        elif request.form.get("action", None) == "Save Meshes":
            model.saveMeshes(latents)
            return redirect(url_for("index", result="Saved 3D model as meshes"))
        
        elif request.form.get("action", None) == "New Session":
            img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], '_.jpg')
            if os.path.exists(render_file_path):
                os.remove(render_file_path)
            response = openai.ChatCompletion.create(
                model=CONVER_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant in 3D modelling."},
                    {"role": "user", "content": 'Forget all the above chat and start a fresh session.'}
                    ]
                )
            return redirect(url_for("index", result=response['choices'][0]['message']['content']))
        
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
            #render_template("index.html", result="Waiting for creating a model and rendering...", model_stage=GENERATING_FROM_IMAGE)

        else:
            prompt = request.form["prompt"]
            response = openai.ChatCompletion.create(
                model=CONVER_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant in 3D modelling."},
                    {"role": "user", "content": prompt}
                    ]
                )
            all_session += '\nUser: ' + prompt + '\nAssistant:' + response['choices'][0]['message']['content']
            return redirect(url_for("index", result=response['choices'][0]['message']['content']))

    # if request.args.get("model_stage") == GENERATING_FROM_TEXT:
    #     prompt_3d = response['choices'][0]['message']['content']
    #     print(prompt_3d)
    #     latents = model.gen_from_text(prompt_3d)
    #     return redirect(url_for("index", result=response['choices'][0]['message']['content'] + "\n3D model generated!", model_stage=IDLE))

    # if request.args.get("model_stage") == GENERATING_FROM_IMAGE:
    #     if os.path.exists(img_file_path):
    #         print("generating from image")
    #         latents = model.gen_from_image(img_file_path)
    #     return redirect(url_for("index", result="3D model generated from input image!", model_stage=IDLE))
    
    ifp = None
    rfp = None
    if os.path.exists(img_file_path):
        ifp = img_file_path
    if os.path.exists(render_file_path):
        rfp = render_file_path
    return render_template("index.html", result=request.args.get("result"), upload_image=ifp, render_3d=rfp)


def generate_3d_prompt(session):
    return """{}\nExtract the most relevant term from your latest response to my most recent prompt and answer with only the term. You can also add any description to your answer if it helps the 3D modelling, such as dimensions and shape ratios, but only the details in your answer""".format(session)
    
    #Answer with only the name of one most favorable answer among possible answers.
    #For example, the expected response to the prompt "what can calculate numerical operations?" is "calculator".
    #Add description to your answer if it is needed to identify the object/concpet being referred in the answer.
    #My prommpt: {}""".format(
    #    prompt #.capitalize()
    #)

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
