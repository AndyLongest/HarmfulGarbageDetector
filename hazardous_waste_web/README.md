# 有害垃圾智能检测 Web 应用

## 检测 Pipeline

上传图片会先调用仓库根目录中的 `preprocess.py`，按照以下流程处理后再进入 YOLOv5：

```text
上传图片 -> Resize -> GaussianBlur -> CLAHE -> Sharpen -> YOLOv5 -> 有害垃圾告警
```

## Windows 一键启动

直接双击仓库根目录中的：

```text
一键启动.bat
```

脚本会自动检查 `YOLO` Conda 环境、启动检测服务，并打开浏览器。关闭启动窗口不会停止检测服务。

需要停止服务时，双击：

```text
停止服务.bat
```

服务启动日志位于 `hazardous_waste_web/server.log` 和
`hazardous_waste_web/server-error.log`。

## 命令行启动

在仓库根目录并激活 `YOLO` 环境后运行：

```powershell
python hazardous_waste_web/app.py
```

浏览器打开 <http://127.0.0.1:8000>。

可选参数：

```powershell
python hazardous_waste_web/app.py --host 0.0.0.0 --port 8000 --device cpu
```

有害垃圾类别配置位于 `hazardous_waste_web/app.py` 中的 `HAZARDOUS_CLASSES`。
