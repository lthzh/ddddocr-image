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


# ==========================================
# 新增接口：滑块拼图缺口匹配 (Slide Match)
# 保留原有 /health 与 /predict 接口完全不变
# ==========================================

class SlideMatchRequest(BaseModel):
    target_base64: str = ""       # 拼图切片小图 Base64 (支持带 data:image/... 前缀)
    background_base64: str = ""   # 带缺口背景图 Base64 (支持带 data:image/... 前缀)
    piece_base64: str = ""        # 兼容字段别名
    bg_base64: str = ""           # 兼容字段别名
    simple_target: bool = True    # 切片图为透明通道切块时推荐为 True


@app.post("/slide_match")
@app.post("/slide")
def slide_match(req: SlideMatchRequest):
    target_str = req.target_base64 or req.piece_base64
    bg_str = req.background_base64 or req.bg_base64

    if not target_str or not bg_str:
        raise HTTPException(
            status_code=400,
            detail={"error": "MISSING_IMAGES", "message": "Both target_base64 and background_base64 are required"}
        )

    # 1. Base64 格式解析与解码
    try:
        if "," in target_str:
            target_str = target_str.split(",", 1)[1]
        if "," in bg_str:
            bg_str = bg_str.split(",", 1)[1]
        target_bytes = base64.b64decode(target_str)
        bg_bytes = base64.b64decode(bg_str)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_BASE64"}
        )

    # 2. 执行滑块缺口匹配
    try:
        res = ocr.slide_match(target_bytes, bg_bytes, simple_target=req.simple_target)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "SLIDE_MATCH_FAILED", "message": str(e)}
        )

    if not res or "target" not in res:
        raise HTTPException(
            status_code=422,
            detail={"error": "GAP_NOT_FOUND"}
        )

    target_box = res["target"]  # [x1, y1, x2, y2]
    return {
        "target": target_box,
        "target_x": target_box[0],
        "target_y": res.get("target_y", target_box[1]),
        "box": {
            "x1": target_box[0],
            "y1": target_box[1],
            "x2": target_box[2],
            "y2": target_box[3]
        }
    }
