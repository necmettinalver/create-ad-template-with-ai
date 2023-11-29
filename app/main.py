from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Request
import io, uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import joblib
import pandas as pd
import torch

from concurrent.futures import ThreadPoolExecutor
#!pip install python-multi- look at the contined verb

#yeni
from diffusers import AutoPipelineForImage2Image#, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid, load_image

current_directory = os.path.dirname(os.path.realpath(__file__))
print(f"-----------------------\n{current_directory}\n--------------------------")
app = FastAPI()
app.POOL: ThreadPoolExecutor = None
templates = Jinja2Templates(directory="app/templates")

data = pd.read_csv('app/ml_color/color_names.csv')
loaded_model = joblib.load('app/ml_color/base_model.joblib')

# cuda or cpu config
def get_device():
    if torch.cuda.is_available():
        print('cuda is available')
        return torch.device('cuda')
    else:
        print('cuda is not available')
        return torch.device('cpu')

"""
@app.on_event("startup")
def startup_event():
    app.POOL = ThreadPoolExecutor(max_workers=1)
@app.on_event("shutdown")
def shutdown_event():
    app.POOL.shutdown(wait=False)
"""

model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "app/stable-diffusion-v1-5"
device = get_device()
pipeline = AutoPipelineForImage2Image.from_pretrained(model_id, safety_checker=None).to(device)#, torch_dtype=torch.float16
#pipeline.save_pretrained(r'app/stable_model')
#! pipeline.enable_sequential_cpu_offload()
#pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
#pipeline = pipeline.to("cuda")


def hex_to_rgb(hex_string):
    # Hex kodunu RGB değerlerine dönüştür
    if hex_string[0]== '#':
        hex_string = hex_string[1:]
    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)
    return r, g, b

def prompt_with_colorname(prompt,hex):
    r,g,b = hex_to_rgb(hex)
    if len(data[(data['Red (8 bit)'] == r) & (data['Green (8 bit)'] == g) & (data['Blue (8 bit)'] == b)]) > 0:
        predicted_color = data.loc[(data['Red (8 bit)'] == r) & (data['Green (8 bit)'] == g) & (data['Blue (8 bit)'] == b), 'Name'].values[0]
        print(f'Veri setinde bulundu. r={r} - g={g} - b={b}')
    else:
        predicted_color = loaded_model.predict([[r,g,b]])[0]
        print(f'yapay zeka ile tahmin ettirildi. r={r} - g={g} - b={b}')

    prompt = prompt + f', {predicted_color}'    
    return prompt


def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, "white")
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

def create_image(logo,created_path,button_path, button_text, punchline_text,color):
    template_shape = 800
    template = Image.new("RGB", (template_shape, template_shape), "white")
    
    pading_y = template_shape // 5  # int(template_shape/4)
    pading_x = int(template_shape / 2)
    
    
    # İlk görseli yükle ve küçült
    #image1 = Image.open(logo).convert('RGBA')
    logo_shape = template_shape//9
    logo = logo.resize((logo_shape, logo_shape))  # İlk görseli istediğiniz boyuta ölçeklendirin
    #image1 = add_corners(image1, 30)
    template.paste(logo, ((pading_x-logo_shape//2), template_shape//25), logo)
  
    # İlk görseli yükle ve küçült
    created_image = Image.open(created_path)
    created_image_shape = template_shape//2
    created_image = created_image.resize((created_image_shape, created_image_shape))  # İlk görseli istediğiniz boyuta ölçeklendirin
    created_image = add_corners(created_image, 30)
    pading_img2 = 2*template_shape//30+logo_shape
    template.paste(created_image, ((pading_x-created_image_shape//2), pading_img2), created_image)

    
    # Button
    button = Image.open(button_path).convert("RGB")
    button_shape_x = int(created_image_shape // 1.5)
    button_shape_y = logo_shape // 2
    button = button.resize((button_shape_x, button_shape_y))
    # Split into 3 channels
    data = button.getdata()
    new_image = []
    for item in data:
        # change all white (also shades of whites)
        # pixels to yellow
        if item[0] in list(range(0, 50)):
            new_image.append((color[0], color[1], color[2]))
        else:
            new_image.append(item)
    
    # update image data
    button.putdata(new_image)
    button = button.convert('RGBA')
    button_size = template_shape//80
    button = add_corners(button, button_size)
    template.paste(button, (pading_x - (button_shape_x // 2), (template_shape-(template_shape//18)-button_shape_y)), button) #15 and down is 15
    # Button text
    draw = ImageDraw.Draw(template)
    button_text_size = 35*button_size/len(button_text)*1.2
    font = ImageFont.truetype(os.path.join(current_directory,"images", "arial_narrow_7.ttf"), button_text_size)
    draw.text((pading_x - (button_shape_x // 2) + (button_shape_x//4 - button_text_size) , (template_shape-(template_shape//18)-button_shape_y + (button_shape_y//1.2 - button_text_size))), button_text, font=font, fill=(255, 255, 255))
    
    # Punchline text
    draw = ImageDraw.Draw(template)
    font = ImageFont.truetype(os.path.join(current_directory,"images", "arial_narrow_7.ttf"), 50)  # Varsayılan font nesnesi
    # Hesaplanan metin boyutu
    text_length = draw.textlength(text=punchline_text, font=font)
    # Eğer metin boyutu template boyutundan büyükse
    if text_length > template_shape:
        # Yeni metin ekleyin
        half_length = len(punchline_text) // 2
        # Yarının sonrasındaki ilk boşluğu bul
        space_index = punchline_text.find(' ', half_length)
        modified_text = punchline_text[:space_index] + '\n' + punchline_text[space_index + 1:]
        half_text_length = draw.textlength(text=punchline_text[:space_index], font=font)
        draw.text((pading_x-half_text_length//2, (template_shape-(template_shape//5)-button_shape_y)), modified_text, font=font, fill=(color[0], color[1], color[2]))
    else:
        # Eğer metin boyutu uygunsa, orijinal metni çizin
        draw.text((pading_x-text_length//2, (template_shape-(template_shape//5)-button_shape_y)), punchline_text, font=font, fill=(color[0], color[1], color[2]))
    
    
    #upperline
    line_width, line_height = template_shape//1.1, 7
    draw = ImageDraw.Draw(template)
    rect_coords = [pading_x-line_width//2, -line_height, pading_x+line_width//2, line_height]  # Sol üst ve sağ alt koordinatları
    corner_radius = 20#pading_x-line_width, template_shape-(template_shape//15)-line_height, pading_x+line_width, template_shape-(template_shape//15)
    draw.rounded_rectangle(rect_coords, corner_radius, fill=(color[0], color[1], color[2]))
    #loewerline
    draw = ImageDraw.Draw(template)
    rect_coords = [pading_x-line_width//2, template_shape-line_height, pading_x+line_width//2, template_shape+line_height]  # Sol üst ve sağ alt koordinatları
    corner_radius = 20#pading_x-line_width, template_shape-(template_shape//15)-line_height, pading_x+line_width, template_shape-(template_shape//15)
    draw.rounded_rectangle(rect_coords, corner_radius, fill=(color[0], color[1], color[2]))
    return template


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("getimg2img.html", {"request": request})

@app.get("/get_output_template", response_class=HTMLResponse)
async def read(request: Request):
    return templates.TemplateResponse("get_template.html", {"request": request})
    
@app.post("/color_name")
async def root(prompt: str = Form("disney pixar"),
    hex: str = Form("#000000")):
    prompt = prompt_with_colorname(prompt,hex)
    return {"message": f"Welcome to the Text-to-Image API! - {prompt}"}
    
@app.post("/getimg2img")
async def get_img2img(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form("disney pixar"),
    hex: str = Form("#000000")):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        filtered_image = io.BytesIO()
        if image.mode in ('RGBA', 'LA'):
            image = image.convert("RGB")
        image = image.resize((512, 512))
        
        prompt = prompt_with_colorname(prompt,hex)
        
        #image = app.POOL.submit(pipeline,image,prompt).result().images
        image_created = pipeline(prompt=prompt, image=image).images[0]
        
        new_image_path = os.path.join(current_directory,"images", "image_createdx.jpg")
        image_created.save(new_image_path)
        image_created.save(filtered_image, 'JPEG')
        filtered_image.seek(0)
        
        #return StreamingResponse(filtered_image, media_type="image/jpeg")
        redirect_url = request.url_for('read')
        return RedirectResponse(url=redirect_url, status_code=303)
    except Exception as e:
        print(f"error: {str(e)}")
        return JSONResponse(content={f"error": str(e)}, status_code=500)    
    
   
@app.post("/output_template")
async def get_img2img(
    request: Request,
    logo: UploadFile = File(...),
    hex: str = Form("#000000"),
    punchline_text: str = Form("AI ad banners lead to higher conversions ratesxxxx"),
    button_text: str = Form("call to action text here")):
    
    r,g,b = hex_to_rgb(hex)
    
    # Gönderilen dosyanın içeriğini oku
    contents = await logo.read()
    logo = Image.open(io.BytesIO(contents)).convert('RGBA')
    
    

    result_image = create_image(logo, os.path.join(current_directory,"images", "image_createdx.jpg"), os.path.join(current_directory,"images", "buttonx.png"), button_text, punchline_text, [r,g,b])
    filtered_result_image = io.BytesIO()
    result_image.save(filtered_result_image, 'JPEG')
    filtered_result_image.seek(0)
    return StreamingResponse(filtered_result_image, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)