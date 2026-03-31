# Guia de Deploy — Raspberry Pi 3 Model B

Este documento detalha todas as mudanças, procedimentos e otimizações
necessárias para rodar o SIGA no **Raspberry Pi 3 Model B** (ARM Cortex-A53,
1.2GHz quad-core, 1GB RAM, Wi-Fi/Bluetooth integrados).

> **Contexto:** Este guia é uma evolução do `pi2_deployment.md`.
> O Pi 3 oferece **~50% mais performance** que o Pi 2, o que abre
> possibilidades que antes eram inviáveis (ex: 2 modelos YOLO, resolução maior).

---

## Comparativo Pi 2 × Pi 3

| Especificação | Raspberry Pi 2 | Raspberry Pi 3 |
|---|---|---|
| **SoC** | BCM2836 | BCM2837 |
| **CPU** | Cortex-A7 (32-bit) 900MHz | Cortex-A53 (64-bit) 1.2GHz |
| **RAM** | 1GB LPDDR2 | 1GB LPDDR2 |
| **Wi-Fi** | ❌ (requer dongle USB) | ✅ 802.11 b/g/n integrado |
| **Bluetooth** | ❌ | ✅ 4.1 (Classic + BLE) |
| **Speedup geral** | Baseline | ~50% mais rápido |
| **Alimentação** | 5V/2A | 5V/2.5A (recomendado) |

**Impacto prático para o SIGA:**
- Inferência YOLO ~50% mais rápida → viável rodar em 320×240 @15fps
- Possível usar 2 modelos YOLO simultâneos com ONNX
- Wi-Fi integrado simplifica o acesso via SSH
- Bluetooth pode ser usado para fones de ouvido (acessibilidade)
- RAM continua sendo 1GB → **mesmas restrições de memória do Pi 2**

---

## 1. Escolha do Sistema Operacional

### 1.1 Raspberry Pi OS Lite (64-bit) — **Recomendado**

O Pi 3 suporta 64-bit, e a versão Lite (sem desktop) economiza RAM.

```bash
# Grave com o Raspberry Pi Imager
# Selecione: Raspberry Pi OS Lite (64-bit) — Bookworm
# Não instale a versão Desktop — ela consome ~200MB de RAM a mais

# Se usar imager pela CLI:
rpi-imager --cli \
    https://downloads.raspberrypi.com/raspios_lite_arm64/images/ \
    /dev/sdX
```

> **⚠️ 64-bit vs 32-bit:** A versão 64-bit do OS é preferível porque
> o Cortex-A53 é nativo ARMv8 (64-bit). Usar 32-bit desperdiça
> ~10-20% de performance em operações de ponto flutuante.

### 1.2 Configuração inicial via `raspi-config`

```bash
sudo raspi-config

# 1. System Options → Hostname → "siga-pi3"
# 2. System Options → Boot / Auto Login → Console Autologin
# 3. Interface Options → SSH → Enable
# 4. Interface Options → Legacy Camera → Enable (se usar câmera CSI)
# 5. Performance Options → GPU Memory → 64 (mínimo para headless)
#    (libera mais RAM para o sistema — padrão é 76MB)
# 6. Localisation → Timezone → America/Sao_Paulo
# 7. Localisation → Locale → pt_BR.UTF-8

sudo reboot
```

---

## 2. Mudanças no `config.py`

### 2.1 Desabilitar janela gráfica (headless)

O Pi 3 não terá monitor conectado na feira. O OpenCV `imshow`
consome CPU desnecessariamente e causa erros sem display.

```python
# Em config.py → DisplayConfig
show_window: bool = False   # ← mudar de True para False
```

### 2.2 Desabilitar EasyOCR

O EasyOCR consome **300–500MB de RAM** — inadmissível para 1GB total.
Mesmo com o Pi 3 sendo mais rápido, a limitação é a **memória**, não a CPU.

```python
# Em config.py → OcrConfig
enabled: bool = False   # ← mudar de True para False
```

### 2.3 Resolução e FPS — Perfil Otimizado para Pi 3

Com o ganho de ~50% em CPU do Pi 3, podemos ser **mais agressivos**
que no Pi 2. Duas opções:

#### Opção A: Conservadora (máxima estabilidade — recomendada para feira)

Mesma config do Pi 2, mas aproveitando o headroom extra para menor latência:

```python
# Em config.py → CameraConfig
width:       int = 320
height:      int = 240
fps_target:  int = 15    # Pi 3 consegue sustentar 15fps com ONNX
buffer_size: int = 1
```

#### Opção B: Alta Performance (se a câmera USB suportar)

Se a câmera for boa e a latência de detecção estiver abaixo de 200ms:

```python
# Em config.py → CameraConfig
width:       int = 480     # Melhor campo de visão
height:      int = 360
fps_target:  int = 12      # Compensar resolução maior
buffer_size: int = 1
```

> **Recomendação:** Use a Opção A para a feira. Teste a Opção B
> somente se tiver tempo para validar estabilidade por >30min.

### 2.4 Modelos YOLO — O Pi 3 Permite Mais Flexibilidade

Com ONNX, o Pi 3 pode rodar **2 modelos simultâneos** se necessário:

#### Para feira (1 modelo — mais estável):

```python
# Em config.py → ModelConfig
urbano:  Path = MODEL_DIR / "best.onnx"    #← modelo principal
geral:   Path = MODEL_DIR / "disabled"     # ← não carrega
esquina: Path = MODEL_DIR / "disabled"     # ← não carrega
```

#### Para demonstração completa (2 modelos):

```python
# Em config.py → ModelConfig
urbano:  Path = MODEL_DIR / "best.onnx"      # ← principal (obstáculos)
geral:   Path = MODEL_DIR / "yolov8n.onnx"   # ← detecção geral leve
esquina: Path = MODEL_DIR / "disabled"        # ← sem esquina (economia)

# Ajustar intervalos para compensar
urbano_frame_interval:  int = 2    # Mais frequente que no Pi 2
esquina_frame_interval: int = 8    # Ignorado se desabilitado
```

### 2.5 Intervalos de Frame — Otimizados para Cortex-A53

O Pi 3 tem mais headroom, então podemos reduzir os intervalos:

```python
# Em config.py → ModelConfig
urbano_frame_interval:  int = 2   # Pi 2: 3 → Pi 3: 2 (mais responsivo)
esquina_frame_interval: int = 4   # Pi 2: 5 → Pi 3: 4
imgsz: int = 320                  # Manter 320 — bom equilíbrio
```

---

## 3. Exportação para ONNX (Obrigatório)

No Pi 3, ONNX é **obrigatório** — não use PyTorch diretamente.
O ganho de performance e economia de memória justificam totalmente.

### 3.1 Exportar no desktop (uma vez, antes do deploy)

```bash
# No desktop com PyTorch + CUDA instalados:
python -c "
from ultralytics import YOLO

# Modelo principal (best.pt → best.onnx)
model = YOLO('modelos/best.pt')
model.export(format='onnx', imgsz=320, half=False, simplify=True)
print('✅ best.onnx exportado')

# Modelo geral leve (opcional — só se quiser 2 modelos)
model2 = YOLO('modelos/yolov8n.pt')
model2.export(format='onnx', imgsz=320, half=False, simplify=True)
print('✅ yolov8n.onnx exportado')
"
```

O `simplify=True` usa o `onnxsim` para otimizar o grafo, reduzindo
latência de inferência em ~10-15% comparado ao ONNX sem simplificação.

### 3.2 Verificar os arquivos exportados

```bash
ls -lh modelos/*.onnx
# Esperado:
#   best.onnx     ~15MB (vs 16MB do .pt)
#   yolov8n.onnx  ~6MB  (vs 6.5MB do .pt)
```

### 3.3 Configurar no Pi 3

```python
# Em config.py → ModelConfig
urbano: Path = MODEL_DIR / "best.onnx"   # ← extensão .onnx
```

O Ultralytics detecta automaticamente o formato e usa `onnxruntime`
em vez de PyTorch. Não é necessário importar `torch` no Pi.

---

## 4. Câmera no Raspberry Pi 3

### 4.1 Câmera USB (mais comum e testada)

```python
# Em vision_engine.py → _ensure_camera_open()
# O código atual já faz VideoCapture(0), que funciona com USB.
# Se não funcionar, force o backend V4L2:
cap = cv2.VideoCapture(CAMERA.device, cv2.CAP_V4L2)
```

### 4.2 Câmera CSI (Módulo Câmera do Raspberry Pi)

Se usar a câmera oficial CSI em vez de USB:

```bash
# 1. Habilitar legacy camera (necessário para OpenCV funcionar)
sudo raspi-config
# → Interface Options → Legacy Camera → Enable
sudo reboot

# 2. Testar
raspistill -o test.jpg
# Se funcionar, o OpenCV consegue acessar via /dev/video0
```

### 4.3 Libcamera (Raspberry Pi OS Bookworm+)

Em versões mais recentes do OS, `libcamera` substitui `v4l2`:

```bash
# Se raspistill não funcionar, use:
libcamera-still -o test.jpg
libcamera-vid -t 5000 -o test.h264

# Para usar com OpenCV, instale o wrapper:
pip install picamera2

# E adapte vision_engine.py para usar Picamera2:
# (ver seção 4.5 abaixo)
```

### 4.4 Verificar permissões

```bash
# Adicione o usuário ao grupo video
sudo usermod -aG video $USER

# Verifique se a câmera aparece
ls /dev/video*

# Teste com v4l2-utils
sudo apt install v4l-utils
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext -d /dev/video0
```

### 4.5 Alternativa: Picamera2 (para câmera CSI no Bookworm)

Se o OpenCV não conseguir abrir a câmera CSI no Bookworm,
crie um wrapper usando Picamera2:

```python
# Em vision_engine.py — substituir cv2.VideoCapture por:
from picamera2 import Picamera2
import cv2
import numpy as np

class PiCameraCapture:
    """Wrapper de compatibilidade Picamera2 → interface OpenCV."""
    def __init__(self, width=320, height=240, fps=15):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def isOpened(self):
        return self.picam2.is_open

    def read(self):
        frame = self.picam2.capture_array()
        # Picamera2 retorna RGB, OpenCV espera BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self):
        self.picam2.stop()

    def set(self, prop_id, value):
        pass  # Props do OpenCV não se aplicam

    def grab(self):
        self.picam2.capture_array()  # Descarta frame
```

> **Nota:** Use `PiCameraCapture` apenas como fallback.
> A câmera USB com OpenCV nativo (`cv2.VideoCapture`) é mais
> simples e testada. Priorize-a.

### 4.6 Buffer da câmera

A câmera USB pode acumular frames antigos. O código já configura
`CAP_PROP_BUFFERSIZE=1`, mas nem todos os drivers suportam.
O `drain_camera_buffer()` em `vision_engine.py` é o fallback.

---

## 5. Conectividade — Wi-Fi e Bluetooth

### 5.1 Wi-Fi (acesso remoto via SSH)

O Pi 3 tem Wi-Fi integrado. Configure antes de levar à feira:

```bash
# Via raspi-config:
sudo raspi-config
# → System Options → Wireless LAN → SSID + senha

# Ou via arquivo (headless setup):
sudo nmcli dev wifi connect "NOME_DA_REDE" password "SENHA"

# Verificar IP:
ip addr show wlan0
# Anotar o IP para SSH
```

### 5.2 SSH (acesso remoto para debug)

```bash
# No seu computador:
ssh pi@<IP_DO_PI3>

# Para iniciar o SIGA remotamente:
cd ~/SIGA && source venv/bin/activate && python3 src/main.py

# Para monitorar logs em tempo real:
tail -f logs/siga.log
```

### 5.3 Bluetooth para fones de ouvido (acessibilidade)

O Bluetooth integrado permite conectar fones sem fio para o usuário
deficiente visual ouvir as detecções do SIGA:

```bash
# Parear um fone Bluetooth:
sudo bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# scan on
# Esperar o dispositivo aparecer...
[bluetooth]# pair XX:XX:XX:XX:XX:XX
[bluetooth]# connect XX:XX:XX:XX:XX:XX
[bluetooth]# trust XX:XX:XX:XX:XX:XX
[bluetooth]# quit

# Definir como saída de áudio padrão:
pactl set-default-sink bluez_sink.XX_XX_XX_XX_XX_XX.a2dp_sink
```

> **Dica para a feira:** Conecte um fone Bluetooth antes de ligar o SIGA.
> Isso demonstra o uso real do produto (deficiente visual com fone ouvindo
> alertas enquanto caminha).

---

## 6. Gerenciamento Térmico

O Pi 3 gera **mais calor** que o Pi 2 por causa do clock maior (1.2GHz).
A carga contínua de inferência YOLO pode causar throttling térmico.

### 6.1 Monitorar temperatura

```bash
# Temperatura atual do CPU:
vcgencmd measure_temp
# Saída: temp=52.0'C (normal até 80°C, throttle em 82°C)

# Monitorar continuamente:
watch -n 2 vcgencmd measure_temp

# Verificar se throttling está ocorrendo:
vcgencmd get_throttled
# 0x0 = OK | 0x50005 = throttling ativo
```

### 6.2 Dissipador e ventilação

```
⚠️  OBRIGATÓRIO para deploy com SIGA:
    Instale um dissipador de calor passivo no SoC (BCM2837).
    Preço: ~R$5-10. Sem ele, throttling é quase garantido
    sob carga contínua de inferência.

    Opcional (mas recomendado para feiras longas):
    Mini-ventoinha 5V colada ao dissipador.
```

### 6.3 Forçar clock máximo (overclocking leve)

Se o throttling não for problema com dissipador:

```bash
# Em /boot/config.txt (ou /boot/firmware/config.txt no Bookworm):
arm_freq=1300          # 1.3GHz (vs 1.2GHz padrão) — overclock leve
over_voltage=2         # +0.050V para estabilidade
gpu_mem=64             # Mínimo para headless

# ⚠️  Só faça overclock se tiver dissipador + ventilação!
```

---

## 7. Instalação de Dependências no Pi 3

### 7.1 Dependências do sistema

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Dependências do sistema
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    portaudio19-dev \
    espeak \
    libatlas-base-dev \
    libjpeg-dev libpng-dev \
    git htop tmux
```

### 7.2 Ambiente virtual Python

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip setuptools wheel
```

### 7.3 Dependências Python

```bash
# Dependências Python (versões otimizadas para ARM64)
pip install \
    opencv-python-headless \
    pyttsx3 \
    vosk \
    sounddevice \
    numpy

# ONNX Runtime para inferência (OBRIGATÓRIO)
pip install onnxruntime

# Ultralytics (necessário para carregar modelos ONNX via API do YOLO)
pip install ultralytics

# NÃO instale PyTorch — economiza ~500MB de RAM/disco
# NÃO instale easyocr — economiza ~300-500MB de RAM
```

> **Nota sobre ARM64:** Se algum pacote falhar na instalação,
> tente instalar via apt (versão do sistema):
> ```bash
> sudo apt install python3-numpy python3-opencv
> ```
> E depois crie o venv com `--system-site-packages`:
> ```bash
> python3 -m venv --system-site-packages venv
> ```

---

## 8. Transferência de Arquivos para o Pi 3

### 8.1 Via SCP (recomendado)

```bash
# No seu computador (PowerShell/Terminal):
# Substituir <IP> pelo IP do Pi 3

# Copiar código-fonte
scp -r C:\SIGA\src pi@<IP>:~/SIGA/src/

# Copiar modelos ONNX (atenção: arquivos grandes!)
scp C:\SIGA\modelos\best.onnx pi@<IP>:~/SIGA/modelos/
scp C:\SIGA\modelos\data.yaml pi@<IP>:~/SIGA/modelos/

# Copiar modelo Vosk
scp -r C:\SIGA\modelos\vosk-model-small-pt-0.3 pi@<IP>:~/SIGA/modelos/
```

### 8.2 Via Git (se o repositório estiver no GitHub)

```bash
# No Pi 3:
cd ~ && git clone https://github.com/Pietro-Freitas/Visao-amiga.git SIGA
cd SIGA
```

### 8.3 Via USB (alternativa offline)

```bash
# 1. Copiar para um pendrive formatado em FAT32 ou ext4
# 2. Montar no Pi:
sudo mount /dev/sda1 /mnt
cp -r /mnt/SIGA ~/SIGA
sudo umount /mnt
```

---

## 9. Script de Inicialização Automática

Para o SIGA iniciar automaticamente quando o Pi 3 ligar:

### 9.1 Criar serviço systemd

```bash
sudo tee /etc/systemd/system/siga.service << 'EOF'
[Unit]
Description=SIGA — Sistema Inteligente de Guiagem Assistiva
After=multi-user.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/SIGA
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/SIGA/venv/bin/python3 src/main.py
Restart=on-failure
RestartSec=10s

# Limites de recursos (evitar OOM)
MemoryMax=850M
MemoryHigh=700M

# Logs
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

### 9.2 Habilitar e testar

```bash
# Recarregar systemd
sudo systemctl daemon-reload

# Habilitar no boot
sudo systemctl enable siga.service

# Testar manualmente
sudo systemctl start siga.service

# Ver status
sudo systemctl status siga.service

# Ver logs
journalctl -u siga.service -f

# Parar
sudo systemctl stop siga.service
```

---

## 10. Checklist de Deploy

### Antes de sair do desktop
- [ ] Exportar modelos para `.onnx` (seção 3)
- [ ] Testar modelos `.onnx` no desktop: `python -c "from ultralytics import YOLO; m=YOLO('modelos/best.onnx'); print(m('test.jpg'))"`
- [ ] Gravar Raspberry Pi OS Lite 64-bit no microSD

### No Raspberry Pi 3
- [ ] Configurar Wi-Fi, SSH, hostname via `raspi-config`
- [ ] Instalar dependências do sistema (seção 7.1)
- [ ] Criar ambiente virtual Python (seção 7.2)
- [ ] Instalar dependências Python (seção 7.3)
- [ ] Copiar código e modelos para `~/SIGA/` (seção 8)

### Configuração do SIGA
- [ ] Alterar `config.py`: `show_window = False` (seção 2.1)
- [ ] Alterar `config.py`: `OCR.enabled = False` (seção 2.2)
- [ ] Alterar `config.py`: usar caminhos `.onnx` (seção 2.4)
- [ ] Verificar `config.py`: resolução 320×240 @15fps (seção 2.3)

### Testes de hardware
- [ ] Testar câmera: `python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened()); c.release()"`
- [ ] Testar microfone: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- [ ] Testar TTS: `python3 -c "import pyttsx3; e=pyttsx3.init(); e.say('teste SIGA'); e.runAndWait()"`
- [ ] Testar modelo ONNX: `python3 -c "from ultralytics import YOLO; m=YOLO('modelos/best.onnx'); print('OK')"`

### Testes de integração
- [ ] Executar: `cd ~/SIGA && source venv/bin/activate && python3 src/main.py`
- [ ] Verificar RAM: `htop` (deve ficar abaixo de ~700MB)
- [ ] Verificar temperatura: `vcgencmd measure_temp` (deve ficar abaixo de 75°C)
- [ ] Teste de estresse: rodar **30 minutos** sem OOM ou throttling
- [ ] Testar comando de voz "encontrar pessoa"
- [ ] Testar detecção de obstáculo → anúncio via TTS

### Preparação para feira
- [ ] Instalar dissipador de calor no SoC
- [ ] Configurar início automático (seção 9)
- [ ] Parear fone Bluetooth (se aplicável, seção 5.3)
- [ ] Testar com fonte 5V/2.5A (alimentação adequada)
- [ ] Preparar powerbank como backup (5V/2.5A, mínimo 10.000mAh)

---

## 11. Estimativa de Uso de RAM — Pi 3

| Componente | Com PyTorch | Com ONNX |
|---|---|---|
| Python + OS (64-bit) | ~120MB | ~120MB |
| OpenCV (headless) | ~50MB | ~50MB |
| PyTorch runtime | ~300MB | — |
| onnxruntime | — | ~60MB |
| 1 modelo YOLO | ~150MB | ~80MB |
| 2 modelos YOLO | ~300MB | ~160MB |
| Vosk (small-pt) | ~50MB | ~50MB |
| pyttsx3 + áudio | ~20MB | ~20MB |
| EasyOCR (se habilitado) | ~400MB | ~400MB |
| **Total (1 modelo, sem OCR)** | **~690MB** | **~380MB** ✅ |
| **Total (2 modelos, sem OCR)** | **~840MB** ⚠️ | **~460MB** ✅ |
| **Total (1 modelo, com OCR)** | **~1090MB** ❌ | **~780MB** ⚠️ |

> ✅ **Com ONNX + 1 modelo + sem OCR = ~380MB** → Sobra ~600MB para o sistema.
> Configuração ideal para feira.
>
> ✅ **Com ONNX + 2 modelos + sem OCR = ~460MB** → Viável no Pi 3 se estável.
>
> ⚠️ **Com ONNX + OCR = ~780MB** → Possível, mas arriscado. Não recomendado.
>
> ❌ **Com PyTorch = RAM insuficiente para operação estável.**

---

## 12. Troubleshooting

### 12.1 "Câmera não abre"

```bash
# Verificar se o dispositivo existe
ls /dev/video*

# Se vazio, testar com:
sudo modprobe bcm2835-v4l2   # Para câmera CSI
# ou reconectar a câmera USB

# Permissões
sudo usermod -aG video $USER
# Deslogar e logar novamente
```

### 12.2 "Microfone não detectado"

```bash
# Listar dispositivos de áudio
arecord -l

# Se vazio, verificar USB
lsusb

# Testar gravação
arecord -d 3 -f cd test.wav
aplay test.wav
```

### 12.3 "OOM Killer matou o processo"

```bash
# Verificar logs
dmesg | grep -i "oom\|killed"

# Soluções:
# 1. Garantir OCR desabilitado
# 2. Usar apenas 1 modelo ONNX
# 3. Reduzir GPU memory: sudo raspi-config → Performance → GPU Memory → 16
# 4. Criar swap (último recurso — degrada performance):
sudo fallocate -l 512M /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 12.4 "Throttling térmico"

```bash
# Verificar
vcgencmd get_throttled
# Se != 0x0, instale dissipador + reduza clock:
# Em /boot/firmware/config.txt:
arm_freq=1100   # Reduzir de 1200 para 1100MHz
```

### 12.5 "TTS não funciona"

```bash
# Testar espeak diretamente
espeak "teste" -v pt

# Se falhar, instalar:
sudo apt install espeak espeak-ng

# Verificar saída de áudio
aplay /usr/share/sounds/alsa/Front_Center.wav

# Forçar saída por 3.5mm (em vez de HDMI):
sudo amixer cset numid=3 1
```

### 12.6 "Vosk não carrega o modelo"

```bash
# Verificar se o modelo existe
ls -la ~/SIGA/modelos/vosk-model-small-pt-0.3/

# Se vazio, baixar:
cd ~/SIGA/modelos
wget https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip
unzip vosk-model-small-pt-0.3.zip
rm vosk-model-small-pt-0.3.zip
```

---

## 13. Dicas para a Feira / Apresentação do TCC

1. **Ligue o Pi 3 com 10 min de antecedência** — o boot + carregamento
   de modelos leva ~30-60 segundos, mas é bom ter margem.

2. **Use uma fonte de qualidade (5V/2.5A)** — fontes fracas causam
   "undervoltage detected" com relâmpago amarelo no log.

3. **Powerbank como backup** — mínimo 10.000mAh com saída 5V/2.5A.
   Teste antes para verificar que sustenta a carga.

4. **Fone Bluetooth** — demonstra o caso de uso real de acessibilidade.
   Pareie antes e teste o TTS no fone.

5. **SSH preparado** — se algo der errado, acesse via SSH do celular
   (apps como Termius/JuiceSSH) para debug rápido.

6. **Vídeo de backup** — grave um vídeo do SIGA funcionando antes da feira.
   Se o hardware falhar no dia, você tem comprovação.

7. **Dissipador visível** — mostra ao avaliador que você considerou
   o gerenciamento térmico (diferencial técnico no TCC).

---

## 14. Estrutura Final de Diretórios no Pi 3

```
~/SIGA/
├── src/
│   ├── config.py           # ← Modificado conforme seção 2
│   ├── main.py
│   ├── model_manager.py
│   ├── speech_engine.py
│   ├── vision_engine.py
│   └── voice_interface.py
├── modelos/
│   ├── best.onnx           # ← Exportado do desktop (seção 3)
│   ├── yolov8n.onnx        # ← Opcional (2 modelos)
│   ├── data.yaml
│   └── vosk-model-small-pt-0.3/
├── logs/                   # ← Criado automaticamente
├── docs/
│   ├── pi2_deployment.md
│   └── pi3_deployment.md   # ← Este arquivo
├── venv/                   # ← Ambiente virtual Python
└── README.md
```
