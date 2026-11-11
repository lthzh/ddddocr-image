# ddddocr-image 多功能验证码识别服务

基于 FastAPI + ddddocr + RapidOCR 构建的轻量、多功能统一验证码识别微服务。

---

## 一、接口列表与功能支持

| 接口路由 | 请求方式 | 对应场景 | 请求体关键参数 |
| :--- | :---: | :--- | :--- |
| **`/health`** | `GET` | 服务健康检查 | 无 |
| **`/predict`** | `POST` | 4位纯数字验证码识别 | `image_base64` |
| **`/slide_match`** (别名 `/slide`) | `POST` | 滑块拼图缺口匹配 | `target_base64`, `background_base64` |
| **`/click_match`** (别名 `/click`, `/solve`) | `POST` | **顺序汉字/图标点选验证码定位** | `image`, `promptImage` |

---

## 二、点选验证码接口说明 (`/click_match` / `/solve`)

### 请求示例
```json
POST /click_match
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "promptImage": "data:image/png;base64,iVBOR..."
}
```

*参数说明*：
* `image`：点选验证码背景大图（300×200），支持带 `data:image/...;base64,` 前缀或纯 Base64；
* `promptImage`：要求依次点击的提示文字图。
* *别名字段支持*：`image_base64` / `bg_base64` 与 `prompt_base64` / `target_base64` 亦可自动识别。

### 响应示例
```json
{
  "success": true,
  "data": [
    { "char": "拒", "x": 75, "y": 100 },
    { "char": "羯", "x": 68, "y": 52 },
    { "char": "咛", "x": 232, "y": 168 }
  ],
  "coordinates": [
    [75, 100],
    [68, 52],
    [232, 168]
  ],
  "chars": ["拒", "羯", "咛"]
}
```

---

## 三、内部双引擎架构

1. **优先引擎（RapidOCR）**：
   * 原生覆盖 6600+ 常用与生僻汉字；
   * 端到端检测文本框并在提示字与背景字之间做语义和顺序比对；
   * 输出高精度中心点击点。
2. **降级引擎（ddddocr 目标检测）**：
   * 若环境中未安装高阶 OCR 模型，自动回退至 `ddddocr(det=True)` 轮廓检测定位切片。
3. **类型防崩溃处理**：
   * 返回给客户端的坐标均转换为标准 Python `int`，防止 FastAPI 出现 `int64 is not JSON serializable` 报错。
