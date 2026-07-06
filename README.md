# Render OCR API

独立 PDF 文本抽取/OCR 后端，供扣子工作流通过 HTTP 调用。

## 目标

这个服务只做后端 OCR/API，不做前端上传页面：

```text
PDF URL -> 创建 job -> 后台处理 -> 查询状态 -> 获取文本结果
```

处理策略：

```text
每页先用 pdftotext 抽文本
有足够文本 -> 直接返回
没有文本 -> 渲染为图片 -> RapidOCR/Tesseract OCR
```

## 接口

### 健康检查

```http
GET /health
```

### 依赖检查

```http
GET /dependencies
```

部署后先调用这个接口确认：

```text
pdftotext / pdftoppm / tesseract / rapidocr
```

### 创建任务

```http
POST /jobs
Content-Type: application/json
Authorization: Bearer <API_TOKEN>

{
  "file_url": "https://example.com/input.pdf",
  "file_name": "input.pdf",
  "max_pages": 400,
  "dpi": 150,
  "langs": "eng+chi_tra",
  "ocr_engine": "auto"
}
```

### 上传 PDF 并创建任务

用于浏览器直接把 PDF 上传到 Render，绕过扣子/Next.js 上传限制和 Coze S3 proxy CORS 问题。

```http
POST /jobs/upload
Content-Type: multipart/form-data
Authorization: Bearer <API_TOKEN>

file=<PDF 文件>
file_name=input.pdf
max_pages=400
dpi=150
langs=eng+chi_tra
ocr_engine=auto
```

返回：

```json
{
  "job_id": "abc123",
  "status": "queued",
  "status_url": "/jobs/abc123",
  "result_url": "/jobs/abc123/result"
}
```

### 查询状态

```http
GET /jobs/{job_id}
```

### 获取结果

```http
GET /jobs/{job_id}/result
```

### 获取纯文本

```http
GET /jobs/{job_id}/text
```

## Render 部署

1. 把 `render_ocr_service/` 作为独立项目推到 GitHub。
2. Render 新建 `Web Service`。
3. 选择 Docker 部署。
4. 设置环境变量：

```text
API_TOKEN=换成你自己的长随机字符串
STORAGE_DIR=/app/storage
MAX_DOWNLOAD_MB=100
MAX_PAGES_PER_JOB=400
DEFAULT_DPI=150
DEFAULT_LANGS=eng+chi_tra
CORS_ALLOW_ORIGINS=*
PDF_TEXT_TIMEOUT_SECONDS=45
PAGE_RENDER_TIMEOUT_SECONDS=90
PAGE_OCR_TIMEOUT_SECONDS=120
TESSERACT_TIMEOUT_SECONDS=120
OCR_WORKER_RECYCLE_PAGES=50
```

5. 部署完成后访问：

```text
https://你的服务.onrender.com/health
https://你的服务.onrender.com/dependencies
```

如果 `/dependencies` 里 `pdftotext` 为空，说明 Poppler 没装成功；如果 `rapidocr=false`，会自动 fallback 到 Tesseract。

## 扣子接入

导入 `coze_openapi.yaml` 前，把：

```text
https://YOUR_RENDER_SERVICE.onrender.com
```

替换成你的 Render 真实域名。

推荐工作流：

```text
用户上传 PDF
 -> 获取文件 URL
 -> POST /jobs
 -> 轮询 GET /jobs/{job_id}
 -> status 为 completed/completed_with_errors 后
 -> GET /jobs/{job_id}/result 或 /text
 -> 交给大模型分析
```

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

macOS 本地还需要：

```bash
brew install poppler tesseract tesseract-lang
```

## 注意

- Render 免费/低配实例不适合 1000 页扫描件长时间 OCR。
- `STORAGE_DIR` 默认是本地磁盘；生产上要长期保存结果时，应使用 Render Disk、数据库或对象存储。
- 扫描件 OCR 结果不是法律/财务意义上的权威文本，重要字段应人工复核。
