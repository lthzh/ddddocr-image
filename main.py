import base64
import io
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import ddddocr

app = FastAPI()

# 1. 初始化字符识别模型单例（保留原有 /predict 兼容性）
ocr = ddddocr.DdddOcr(show_ad=False)
ocr.set_ranges("0123456789")

# 2. 初始化专用于滑块缺口匹配的模型单例（关闭文字与目标检测以极大提升匹配效率）
slide_engine = ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)

# 3. 初始化目标检测引擎（用于点选验证码备用定位）
det_engine = ddddocr.DdddOcr(det=True, show_ad=False)

# 4. 初始化中文高精度轻量 OCR 模型（用于点选验证码汉字识别与坐标定位）
try:
    from rapidocr_onnxruntime import RapidOCR
    rapid_engine = RapidOCR()
except Exception:
    rapid_engine = None


class CaptchaRequest(BaseModel):
    image_base64: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": {
            "predict": True,
            "slide_match": True,
            "click_match": True,
            "rapidocr_enabled": rapid_engine is not None
        }
    }


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
# 接口：滑块拼图缺口匹配 (Slide Match)
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


# ==========================================
# 新增接口：点选验证码识别与坐标定位 (Click Match)
# 用于 ChmlFrp / Qzhua Captcha 等顺序点选验证码
# ==========================================

class ClickMatchRequest(BaseModel):
    image: str = ""              # 背景大图 Base64 (支持带 data:image/... 前缀)
    promptImage: str = ""        # 提示文字图 Base64 (支持带 data:image/... 前缀)
    image_base64: str = ""       # 别名兼容
    prompt_base64: str = ""      # 别名兼容
    bg_base64: str = ""          # 别名兼容
    target_base64: str = ""      # 别名兼容


@app.post("/click_match")
@app.post("/click")
@app.post("/solve")
def click_match(req: ClickMatchRequest):
    img_str = req.image or req.image_base64 or req.bg_base64
    prompt_str = req.promptImage or req.prompt_base64 or req.target_base64

    if not img_str or not prompt_str:
        raise HTTPException(
            status_code=400,
            detail={"error": "MISSING_IMAGES", "message": "Both image and promptImage are required"}
        )

    # 1. Base64 格式解析与解码
    try:
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]
        if "," in prompt_str:
            prompt_str = prompt_str.split(",", 1)[1]
        img_bytes = base64.b64decode(img_str)
        prompt_bytes = base64.b64decode(prompt_str)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_BASE64"}
        )

    coordinates = []

    # 2. 优先策略：使用 RapidOCR 进行端到端高精度中文点选匹配
    if rapid_engine is not None:
        try:
            # 2.1 识别提示文字中的目标汉字序列
            prompt_clean = normalize_image_bytes(prompt_bytes)
            prompt_res, _ = rapid_engine(prompt_clean)
            target_chars = []
            if prompt_res:
                for item in prompt_res:
                    # 将提取的文本切分为独立单字，如 "拒羯咛" -> ['拒', '羯', '咛']
                    cleaned_txt = re.sub(r"\s+", "", item[1] or "")
                    target_chars.extend(list(cleaned_txt))

            # 2.2 识别背景大图中的所有候选文字及多边形边界框
            bg_clean = normalize_image_bytes(img_bytes)
            bg_res, _ = rapid_engine(bg_clean)

            if target_chars and bg_res:
                # 2.3 按提示文字的先后顺序，在背景大图识别结果中定位中心坐标
                matched_boxes = set()
                for target in target_chars:
                    found = False
                    for idx, (bbox, text, score) in enumerate(bg_res):
                        if idx in matched_boxes:
                            continue
                        if target in text:
                            pts_x = [float(pt[0]) for pt in bbox]
                            pts_y = [float(pt[1]) for pt in bbox]
                            cx = int(round(sum(pts_x) / len(pts_x)))
                            cy = int(round(sum(pts_y) / len(pts_y)))
                            coordinates.append({
                                "char": target,
                                "x": cx,
                                "y": cy
                            })
                            matched_boxes.add(idx)
                            found = True
                            break
        except Exception:
            coordinates = []

    # 3. 降级备用策略：若未安装 RapidOCR 或未能匹配出坐标，调用 ddddocr 目标检测引擎定位候选框
    if not coordinates:
        try:
            bg_clean = normalize_image_bytes(img_bytes)
            bboxes = det_engine.detection(bg_clean)
            if bboxes:
                for box in bboxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    coordinates.append({
                        "char": "",
                        "x": cx,
                        "y": cy
                    })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "CLICK_DETECTION_FAILED", "message": str(e)}
            )

    if not coordinates:
        raise HTTPException(
            status_code=422,
            detail={"error": "COORDINATES_NOT_FOUND", "message": "未能识别出有效的目标点击坐标"}
        )

    # 显式确保所有数值为 Python 原生 int（防 FastAPI Starlette JSON 序列化崩溃）
    data = []
    coord_list = []
    for c in coordinates:
        cx = int(c["x"])
        cy = int(c["y"])
        char_val = str(c.get("char", ""))
        data.append({"char": char_val, "x": cx, "y": cy})
        coord_list.append([cx, cy])

    return {
        "success": True,
        "data": data,
        "coordinates": coord_list,
        "chars": [d["char"] for d in data if d["char"]]
    }
