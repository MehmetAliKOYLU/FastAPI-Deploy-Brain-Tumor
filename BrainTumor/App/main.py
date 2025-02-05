from ultralytics import YOLO
import os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import uvicorn
import numpy as np

model = './App/last.pt'

if os.path.exists(model):
    print(f"Model found: {model}")

app = FastAPI(title= 'YOLO Brain Tumor App')

@app.get('/')
def home():
    return {'message': 'Welcome to the YOLO Brain Tumor Detection App!'}

@app.post('/predict')
async def predict_tumor(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_np = np.array(image)

    yolo = YOLO(model=model)

    results = yolo(image_np)

    # Process results 
    if results:
        result = results[0]
        result.show()  # Display the result
        result.save(filename="result.jpg")  # Save to disk
        return {'message': 'Successfully saved prediction -> result.jpg'}
    else:
        return {'message': 'No detection results.'}

    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

