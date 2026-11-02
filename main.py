import base64
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ddddocr

app = FastAPI()

# 初始化模型单例
ocr = ddddocr.DdddOcr(show_ad=False)

# 强制限制字符集只输出 0-9
ocr.set_ranges("0123456789")

class CaptchaRequest(BaseModel):
    image_base64: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: CaptchaRequest):
    # 1. Base64 格式解析与解码
    try:
        img_str = req.image_base64
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]
        img_bytes = base64.b64decode(img_str)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_BASE64"}
        )

    # 2. 模型推理
    try:
        raw_code = ocr.classification(img_bytes)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"error": "OCR_FAILED"}
        )

    # 3. 提取纯数字并校验长度
    digits = re.sub(r"\D", "", raw_code or "")
    if len(digits) < 4:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INSUFFICIENT_DIGITS",
                "length": len(digits)
            }
        )

    # 正常返回前 4 位识别结果
    return {
        "code": digits[:4]
    }
