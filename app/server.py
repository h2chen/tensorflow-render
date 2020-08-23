import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
#from tensorflow.keras.utils.data_utils import get_file
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


#set url
# export_file_url = 'https://drive.google.com/uc?export=download&id=1ZZ_2JRe39KcgqGu75watpeLOtQGfeDPA'
model_config_name = 'app/models/model.config'
model_file_name = 'app/models/best_model.h5'

classes = ['0', '1', '2', '3']
path = Path(__file__).parent
img_size = 224
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    # await download_file(export_file_url, path / export_file_name)
    try:
        #learn = load_learner(path, export_file_name)        
        #learn = keras.models.load_model("app/"+export_file_name)
        with open(model_config_name, "r") as text_file:
            json_string = text_file.read()
        learn = keras.models.model_from_json(json_string)
        learn.load_weights(model_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
#     img = open_image(BytesIO(img_bytes))   
#     prediction = learn.predict(img)[0]
#     image = tf.keras.preprocessing.image.load_img( path, target_size=(img_size, img_size))
#     input_arr = keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])  # Convert single image to a batch.
#     predictions = learn.predict(input_arr) 

    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((img_size, img_size), Image.NEAREST)
    img = np.array(img)
    img = preprocess_input( np.array([img]) )
    predictions = learn.predict(img)  
    prediction = predictions.argmax()
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5001, log_level="info")
