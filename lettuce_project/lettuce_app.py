from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("lettuce_project/best.pt")

# Request model
class ImageRequest(BaseModel):
    image: str

# Response model
class ClassificationResponse(BaseModel):
    detected: bool
    classification: str = None
    confidence: float = None
    error: str = None

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ImageRequest):
    try:
        # Remove data:image/png;base64, prefix
        image_data = request.image.split(',')[1] if ',' in request.image else request.image
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO prediction
        results = model.predict(image, conf=0.5)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = results[0].names[cls_id]
            
            return ClassificationResponse(
                detected=True,
                classification=label,
                confidence=conf
            )
        else:
            return ClassificationResponse(detected=False)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Lettuce Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
