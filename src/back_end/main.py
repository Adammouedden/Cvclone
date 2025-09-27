from fastapi import FastAPI
from model.data_preprocess import get_data
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()


@app.get('/test')
async def test():
    return {"message": "hello world"}

@app.get('/image')
async def get_image():
    image = get_data.send_image_test()
    return image

@app.get('/image2')
async def get_image2():
    image = get_data.reconstruct_image()
    return image

app.mount(
    "/images",
    StaticFiles(directory="model/data_preprocess/shards"),
    name="images",
)

BASE = Path("model/data_preprocess/shards")

@app.get("/image-list")
def image_list():
    return [f"/images/{p.name}" for p in BASE.glob("*.png")]