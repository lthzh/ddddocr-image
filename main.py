import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ddddocr

app = FastAPI()

# 全局初始化单例：避免每次请求重复加载模型导致 CPU 飙升和延迟
ocr = ddddocr.DdddOcr(show_ad=False)

# 如果你的验证码只有数字，可以加上这行限制以提高准确度：
# ocr.set_ranges("0123456789")

class CaptchaRequest(BaseModel):
    image_base64: str  # 接收 base64 字符串

@app.get("/health")
def health():
    """用于签到脚本在执行前探测并唤醒容器"""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: CaptchaRequest):
    try:
        # 去掉前端可能带有的 data:image/png;base64, 前缀
        img_str = req.image_base64
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]
            
        img_bytes = base64.b64decode(img_str)
        code = ocr.classification(img_bytes)
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
