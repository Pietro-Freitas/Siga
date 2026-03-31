# Guia de Deploy — Raspberry Pi 2

Este documento lista todas as mudanças necessárias para rodar o SIGA
no Raspberry Pi 2 (ARM Cortex-A7, 900MHz, 1GB RAM, headless / só áudio).

> **Importante:** Aplique as mudanças listadas abaixo ANTES de rodar no Pi 2.
> O código atual está configurado para desenvolvimento em desktop.

> **🔗 Possui um Raspberry Pi 3?** Consulte [pi3_deployment.md](pi3_deployment.md)
> para um guia otimizado com Wi-Fi, Bluetooth, inicialização automática e
> configurações que aproveitam o ganho de ~50% de CPU do Cortex-A53.

---

## 1. Mudanças no `config.py`

### 1.1 Desabilitar janela gráfica (headless)

O Pi 2 não terá monitor conectado. O OpenCV `imshow` consome CPU
desnecessariamente e pode causar erros sem display.

```python
# Em config.py → DisplayConfig
show_window: bool = False   # ← mudar de True para False
```

### 1.2 Desabilitar EasyOCR

O EasyOCR consome **300–500MB de RAM** — inadmissível para 1GB total.
Só habilite se for essencial para o uso demonstrado na feira.

```python
# Em config.py → OcrConfig
enabled: bool = False   # ← mudar de True para False
```

### 1.3 Ajustar resolução e FPS (já otimizado)

As configs atuais já estão em 320×240 @15fps, que é o ideal para Pi 2.
Se a câmera suportar, 160×120 pode dar mais performance ainda:

```python
# Em config.py → CameraConfig (apenas se necessário)
width:  int = 160
height: int = 120
fps_target: int = 10
```

### 1.4 Usar apenas 1 modelo YOLO

Com 1GB de RAM, cada modelo YOLO ocupa ~100-200MB. Use apenas 1:

```python
# Em config.py → ModelConfig
# Coloque caminhos "inexistentes" para os modelos que não quer carregar.
# O ModelManager já trata modelos ausentes com best-effort.
urbano:  Path = MODEL_DIR / "best.pt"      # ← seu modelo principal
geral:   Path = MODEL_DIR / "disabled"     # ← inexistente = não carrega
esquina: Path = MODEL_DIR / "disabled"     # ← inexistente = não carrega
```

---

## 2. Exportação para ONNX (Recomendado — ~2× speedup)

ONNX (Open Neural Network Exchange) é um formato de modelo leve que
não precisa do PyTorch completo, economizando **~300MB de RAM**.

### 2.1 Exportar no desktop (uma vez)

```bash
# No desktop com PyTorch instalado:
python -c "
from ultralytics import YOLO
model = YOLO('modelos/best.pt')
model.export(format='onnx', imgsz=320, half=False)
"
```

Isso cria `modelos/best.onnx`.

### 2.2 Usar no Pi 2

```python
# Em config.py → ModelConfig
urbano: Path = MODEL_DIR / "best.onnx"   # ← extensão .onnx
```

O Ultralytics detecta automaticamente o formato e usa `onnxruntime`
em vez de PyTorch. Instale no Pi 2:

```bash
pip install onnxruntime
# Em vez de: pip install torch ultralytics (muito pesado)
```

---

## 3. Problema da Câmera no Pi 2 com OpenCV

Se o OpenCV não consegue abrir a câmera no Pi 2, tente:

### 3.1 Usar backend V4L2 explicitamente

```python
# Em vision_engine.py → _ensure_camera_open()
cap = cv2.VideoCapture(CAMERA.device, cv2.CAP_V4L2)
```

### 3.2 Verificar permissões

```bash
# Adicione o usuário ao grupo video
sudo usermod -aG video $USER

# Verifique se a câmera aparece
ls /dev/video*

# Teste sem OpenCV
v4l2-ctl --list-devices
```

### 3.3 Usar libcamera (Raspberry Pi OS moderno)

Raspberry Pi OS Bullseye+ usa `libcamera` em vez de `v4l2`.
Nesse caso, instale o suporte a legacy camera:

```bash
sudo raspi-config
# → Interface Options → Legacy Camera → Enable
sudo reboot
```

Ou use o backend PiCamera2:

```bash
pip install picamera2
```

E adapte o `vision_engine.py` para usar `Picamera2` em vez de `cv2.VideoCapture`.

### 3.4 Buffer da câmera

A câmera USB pode acumular frames antigos. O código já configura
`CAP_PROP_BUFFERSIZE=1`, mas nem todos os drivers suportam.
O `drain_camera_buffer()` em `vision_engine.py` é o fallback.

---

## 4. Instalação de Dependências no Pi 2

```bash
# Sistema operacional
sudo apt update && sudo apt install -y \
    python3-pip python3-venv \
    libopencv-dev python3-opencv \
    portaudio19-dev \
    espeak

# Ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Dependências Python (versões leves para ARM)
pip install --upgrade pip
pip install \
    opencv-python-headless \
    pyttsx3 \
    vosk \
    sounddevice \
    numpy

# Se usar ONNX (recomendado):
pip install onnxruntime ultralytics

# Se usar PyTorch (pesado, ~500MB):
# pip install torch ultralytics
```

> **Nota:** Use `opencv-python-headless` no Pi 2 (sem suporte GUI).
> Isso economiza ~50MB e evita problemas com display.

---

## 5. Checklist de Deploy

- [ ] Copiar código atualizado para o Pi 2
- [ ] Copiar modelo(s) `.pt` ou `.onnx` para `modelos/`
- [ ] Copiar modelo Vosk para `modelos/vosk-model-small-pt-0.3/`
- [ ] Alterar `config.py` conforme seções 1.1-1.4 acima
- [ ] Instalar dependências (seção 4)
- [ ] Testar câmera: `python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened())"`
- [ ] Testar microfone: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- [ ] Testar TTS: `python3 -c "import pyttsx3; e=pyttsx3.init(); e.say('teste'); e.runAndWait()"`
- [ ] Executar: `cd SIGA && python3 src/main.py`
- [ ] Verificar RAM: `htop` (deve ficar abaixo de ~800MB)
- [ ] Teste de estresse: rodar 10 minutos sem OOM

---

## 6. Estimativa de Uso de RAM

| Componente | Com PyTorch | Com ONNX |
|-----------|------------|---------| 
| Python + OS | ~100MB | ~100MB |
| OpenCV (headless) | ~50MB | ~50MB |
| PyTorch runtime | ~300MB | — |
| onnxruntime | — | ~50MB |
| 1 modelo YOLO | ~150MB | ~80MB |
| Vosk (small-pt) | ~50MB | ~50MB |
| pyttsx3 + áudio | ~20MB | ~20MB |
| EasyOCR (se habilitado) | ~400MB | ~400MB |
| **Total (sem OCR)** | **~670MB** | **~350MB** |
| **Total (com OCR)** | **~1070MB** ⚠️ | **~750MB** |

> Com ONNX + sem OCR, o sistema cabe confortavelmente no Pi 2 (1GB).
> Com PyTorch + OCR, há risco real de OOM.
