import base64
import io
import gc
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import ddddocr

app = FastAPI()

# 采用惰性单例模式 (Lazy Singleton) 按需初始化模型，大幅降低常驻内存，杜绝 1C1G 小内存服务器 OOM 崩溃
_ocr = None
_slide_engine = None
_det_engine = None
_rapid_engine = None
_rapid_checked = False


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = ddddocr.DdddOcr(show_ad=False)
        _ocr.set_ranges("0123456789")
    return _ocr


def get_slide_engine():
    global _slide_engine
    if _slide_engine is None:
        _slide_engine = ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)
    return _slide_engine


def get_det_engine():
    global _det_engine
    if _det_engine is None:
        _det_engine = ddddocr.DdddOcr(det=True, show_ad=False)
    return _det_engine


def get_rapid_engine():
    global _rapid_engine, _rapid_checked
    if not _rapid_checked:
        _rapid_checked = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_engine = RapidOCR()
        except Exception:
            _rapid_engine = None
    return _rapid_engine


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
            "rapidocr_enabled": get_rapid_engine() is not None
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

    # 2. 模型推理 (按需惰性加载)
    try:
        raw_code = get_ocr().classification(img_bytes)
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

def normalize_image_bytes(img_bytes: bytes, target_format: str = "PNG", upscale_small: bool = False) -> bytes:
    """确保图片字节流被规范化并处理透明通道（将 RGBA 透明背景转换为纯白底，防止黑字淹没）"""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        # 若包含透明通道 (RGBA 或 LA)，将其合成到纯白底上
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        # 对小尺寸提示条 (如 132x44) 放大 2 倍以大幅提高小字检出率
        if upscale_small and (im.height < 60 or im.width < 180):
            im = im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, format=target_format)
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
        res = get_slide_engine().slide_match(target_clean, bg_clean, simple_target=req.simple_target)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "SLIDE_MATCH_FAILED", "message": str(e)}
        )
    finally:
        gc.collect()

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
    prompt: str = ""             # 提示汉字文本 (例如 "拒,羯,咛" 或 "拒羯咛")
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
    prompt_text = (req.prompt or "").strip()

    if not img_str:
        raise HTTPException(
            status_code=400,
            detail={"error": "MISSING_IMAGE", "message": "Background image is required"}
        )

    if not prompt_str and not prompt_text:
        raise HTTPException(
            status_code=400,
            detail={"error": "MISSING_PROMPT", "message": "Either promptImage or prompt text is required"}
        )

    # 1. Base64 格式解析与解码
    try:
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]
        img_bytes = base64.b64decode(img_str)

        prompt_bytes = b""
        if prompt_str:
            if "," in prompt_str:
                prompt_str = prompt_str.split(",", 1)[1]
            prompt_bytes = base64.b64decode(prompt_str)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_BASE64"}
        )

    coordinates = []
    target_chars = []

    # 1.1 若请求直接携带了目标汉字文本，先过滤 "请依次点击："、"请点击：" 等纯标题前缀
    clean_prompt = re.sub(r"^(请依次点击[：:]?|请点击[：:]?|依次点击[：:]?)", "", prompt_text).strip()
    if clean_prompt:
        target_chars = [c for c in re.split(r"[,，、\s]+", clean_prompt) if c]

    rapid_engine = get_rapid_engine()

    try:
        # 2. 优先策略：使用 RapidOCR 进行端到端高精度中文点选匹配
        if rapid_engine is not None:
            try:
                # 2.1 若未直接提取出有效汉字，则识别提示文字图片中的汉字序列 (2倍放大白底，大幅提升检测率)
                if not target_chars and prompt_bytes:
                    prompt_clean = normalize_image_bytes(prompt_bytes, upscale_small=True)
                    prompt_res, _ = rapid_engine(prompt_clean)
                    if prompt_res:
                        for item in prompt_res:
                            cleaned_txt = re.sub(r"[\s,，、]+", "", item[1] or "")
                            target_chars.extend(list(cleaned_txt))

                # 2.2 识别背景大图中的所有候选文字及多边形边界框
                bg_clean = normalize_image_bytes(img_bytes)
                bg_res, _ = rapid_engine(bg_clean)

                if target_chars and bg_res:
                    matched_boxes = set()
                    # 2.3.1 第一轮：文本精确包含匹配
                    for target in target_chars:
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
                                break

                    # 2.3.2 第二轮补全：如果匹配到的坐标数量少于目标汉字数 (部分变形/旋转字未精确匹配)，从背景大图剩余候选框中按置信度补齐
                    target_count = len(target_chars) if target_chars else 3
                    if len(coordinates) < target_count:
                        remaining_boxes = [
                            (idx, bbox, score) for idx, (bbox, text, score) in enumerate(bg_res)
                            if idx not in matched_boxes
                        ]
                        remaining_boxes.sort(key=lambda x: x[2], reverse=True)
                        needed = target_count - len(coordinates)
                        for idx, bbox, score in remaining_boxes[:needed]:
                            pts_x = [float(pt[0]) for pt in bbox]
                            pts_y = [float(pt[1]) for pt in bbox]
                            cx = int(round(sum(pts_x) / len(pts_x)))
                            cy = int(round(sum(pts_y) / len(pts_y)))
                            coordinates.append({
                                "char": "",
                                "x": cx,
                                "y": cy
                            })
                            matched_boxes.add(idx)
            except Exception:
                coordinates = []

        # 3. 降级备用策略：若未安装 RapidOCR 或坐标仍不足 3 个，调用 ddddocr 目标检测引擎定位候选框补齐
        target_count = len(target_chars) if target_chars else 3
        if len(coordinates) < target_count:
            try:
                bg_clean = normalize_image_bytes(img_bytes)
                bboxes = get_det_engine().detection(bg_clean)
                if bboxes:
                    needed = target_count - len(coordinates)
                    for box in bboxes[:needed]:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        coordinates.append({
                            "char": "",
                            "x": cx,
                            "y": cy
                        })
            except Exception as e:
                if not coordinates:
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
    finally:
        gc.collect()
