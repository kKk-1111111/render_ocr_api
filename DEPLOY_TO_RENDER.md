# Deploy To Render

## 你现在要做的下一步

当前本地已经有可部署项目：

```text
render_ocr_service/
```

下一步是把这个目录作为一个 GitHub 仓库部署到 Render。

## 1. 推到 GitHub

推荐做成独立仓库，例如：

```text
render-ocr-api
```

仓库根目录应该直接长这样：

```text
app.py
Dockerfile
requirements.txt
render.yaml
coze_openapi.yaml
README.md
DEPLOY_TO_RENDER.md
```

不要让仓库根目录变成：

```text
some-repo/render_ocr_service/app.py
```

除非你在 Render 里额外配置 Dockerfile 路径。

## 2. Render 控制台操作

1. 打开 Render Dashboard。
2. 点 `New`。
3. 选择 `Web Service`。
4. 连接 GitHub 仓库。
5. 选择 `render-ocr-api` 仓库。
6. Runtime/Environment 选择 Docker。
7. Health Check Path 填：

```text
/health
```

8. 设置环境变量：

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

9. 点击创建并等待部署。

## 3. 部署后必须验证

先打开：

```text
https://你的服务.onrender.com/health
```

应该返回：

```json
{"ok":true,"app":"Render OCR API"}
```

再打开：

```text
https://你的服务.onrender.com/dependencies
```

至少应该看到：

```json
{
  "pdftotext": "...",
  "pdftoppm": "...",
  "tesseract": "...",
  "rapidocr": true
}
```

如果 `rapidocr` 是 `false`，但 `tesseract` 有路径，服务仍可运行，只是扫描件会慢。

如果 `pdftotext` 和 `tesseract` 都为空，部署不可用，要先看 Render Build Logs。

## 4. 创建测试任务

把下面的 URL 换成你的 Render 服务和一个公网 PDF URL：

```bash
curl -X POST "https://你的服务.onrender.com/jobs" \
  -H "Authorization: Bearer 你的_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "https://example.com/test.pdf",
    "file_name": "test.pdf",
    "max_pages": 5,
    "ocr_engine": "auto"
  }'
```

返回里会有：

```json
{"job_id":"..."}
```

然后查状态：

```bash
curl -H "Authorization: Bearer 你的_API_TOKEN" \
  "https://你的服务.onrender.com/jobs/JOB_ID"
```

完成后取纯文本：

```bash
curl -H "Authorization: Bearer 你的_API_TOKEN" \
  "https://你的服务.onrender.com/jobs/JOB_ID/text"
```

## 5. 扣子接入

把 `coze_openapi.yaml` 里的：

```text
https://YOUR_RENDER_SERVICE.onrender.com
```

替换成真实 Render 域名。

然后在扣子中导入/配置 OpenAPI 工具。

工作流应该是：

```text
用户上传 PDF
 -> 扣子取得 PDF URL
 -> POST /jobs
 -> 轮询 GET /jobs/{job_id}
 -> GET /jobs/{job_id}/text
 -> 大模型分析
```

## 6. 关键限制

- 这个版本接收的是 `file_url`，不是浏览器直接上传文件。
- 扣子提供的 PDF URL 必须能被 Render 访问。
- Render 免费/低配实例不适合长时间处理 1000 页扫描件。
- 当前 job 状态存在本地 `STORAGE_DIR`，服务重启后可能丢失；生产长期版应接数据库或对象存储。
