from fastapi import FastAPI , UploadFile, File , HTTPException
from api.load_model import predict_image
import time

app=FastAPI(title="Plant Disease Detection API")

@app.get("/")
def root():
    return {"status":"API is running "}



@app.post("/predict")
async def prediction(image: UploadFile= File(...)):
    start_time = time.time()
    if not image.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(
            status_code= 400,
            detail=f'invlaiid file type. Expected .jpg or .png got {image.filename}'
        )
    
    try:
        image_bytes=await image.read()
        result=predict_image(image_bytes)

        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # convert to ms


        return {
            "prediction": result,
            "response_time_ms": round(response_time, 2)
        }
        


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"

        )
    
    


