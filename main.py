import base64
import io
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import ddddocr

app = FastAPI()

# 初始化字符识别模型单例（保留原有 /predict 兼容性）
ocr = ddddocr.DdddOcr(show_ad=False)
ocr.set_ranges("0123456789")

# 初始化专用于滑块缺口匹配的模型单例（关闭文字与目标检测以极大提升匹配效率）
slide_engine = ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)

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

def normalize_image_bytes(img_bytes: bytes) -> bytes:
    """确保图片字节流被规范化为标准 PNG 格式（自动兼容 WebP、JPEG 等特殊格式）"""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return img_bytes


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

    # 2. 图像格式兼容性规范化（转换 WebP 为 PNG）并执行滑块缺口匹配
    try:
        target_clean = normalize_image_bytes(target_bytes)
        bg_clean = normalize_image_bytes(bg_bytes)
        res = slide_engine.slide_match(target_clean, bg_clean, simple_target=req.simple_target)
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

    # 关键修复：ddddocr 底层返回值为 NumPy 标量 (np.int32/np.int64)
    # FastAPI 在进行 JSON 响应序列化时无法识别 np.int64，会引发 "TypeError: Object of type int64 is not JSON serializable"
    # 该错误发生在路由执行完毕后的 Starlette 中间件层，导致向客户端返回 500 Internal Server Error
    # 因此必须显式将所有数值转换回 Python 原生 int
    raw_target = res["target"]
    target_box = [int(v) for v in raw_target]
    target_x = int(res.get("target_x", target_box[0]))
    target_y = int(res.get("target_y", target_box[1] if len(target_box) > 1 else 0))

    x1 = target_box[0]
    y1 = target_box[1] if len(target_box) > 1 else target_y
    x2 = target_box[2] if len(target_box) > 2 else x1
    y2 = target_box[3] if len(target_box) > 3 else y1

    return {
        "target": target_box,
        "target_x": target_x,
        "target_y": target_y,
        "box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    }
