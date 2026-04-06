"""
config.py — Configurações Centralizadas do SIGA
================================================
Todas as constantes, caminhos e parâmetros de tunagem ficam aqui.
Alterar um único arquivo reflete em todo o sistema.

NOTA PARA DEPLOY NO RASPBERRY PI:
  Consulte docs/pi2_deployment.md (Pi 2) ou docs/pi3_deployment.md (Pi 3)
  para ajustes específicos de hardware.
  As configs abaixo já estão otimizadas para máxima performance em ARM.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos base
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR.parent / "modelos"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    urbano:  Path = MODEL_DIR / "best.pt"
    geral:   Path = MODEL_DIR / "yolov8n.pt"
    esquina: Path = MODEL_DIR / "esquina.pt"
    vosk:    Path = MODEL_DIR / "vosk-model-small-pt-0.3"

    # Thresholds de confiança (valores mais altos = menos falsos positivos,
    # porém menos detecções; ideal para feiras com ambiente controlado)
    conf_urbano:  float = 0.50
    conf_geral:   float = 0.55
    conf_esquina: float = 0.50

    # Tamanho de imagem para inferência YOLO.
    # Menor = mais rápido. 320 é o mínimo viável; 640 é o padrão do YOLO.
    # PARA PI 2/3: use 320. Em desktop com GPU: pode usar 640.
    imgsz: int = 320

    # Orquestração: a cada quantos frames rodar cada modelo especialista.
    # Quanto maior o intervalo, menos CPU é consumida.
    #   Dica: no Pi 2, usar intervalos maiores (ex: urbano=4, esquina=8)
    urbano_frame_interval:  int = 3   # roda a cada 3 frames
    esquina_frame_interval: int = 5   # roda a cada 5 frames

    # Classes a ignorar no modelo geral (índices numéricos do COCO dataset)
    # Exemplos: 12=stop sign, 24=backpack — ajuste conforme seu dataset
    ignored_classes_geral: tuple = (12, 24)

MODEL = ModelConfig()

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    device:      int   = 0
    width:       int   = 320     # Reduzido de 640 para performance
    height:      int   = 240     # Reduzido de 480 para performance
    fps_target:  int   = 15      # Reduzido de 30 — realista para Pi 2/3
    buffer_size: int   = 1       # Evita acúmulo de frames antigos na câmera
    # Tempo (s) para tentar reconectar após falha
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10

CAMERA = CameraConfig()

# ---------------------------------------------------------------------------
# Áudio / TTS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoiceConfig:
    volume:  float = 1.0
    rate:    int   = 175      # palavras por minuto
    # Anti-spam: intervalo mínimo (s) entre repetições da MESMA detecção
    spam_interval: float = 4.0
    # Tamanho máximo da fila de TTS (descarta mensagens antigas se lotada)
    tts_queue_maxsize: int = 5
    # Limpeza do cache anti-spam a cada N segundos
    spam_cleanup_interval: float = 30.0

VOICE = VoiceConfig()

# ---------------------------------------------------------------------------
# STT (Reconhecimento de Voz)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SttConfig:
    samplerate: int = 16000
    blocksize:  int = 8000
    dtype:      str = "int16"
    channels:   int = 1

STT = SttConfig()

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OcrConfig:
    # Habilitar/desabilitar OCR globalmente.
    # EasyOCR é MUITO pesado (~300-500MB RAM). Desabilite no Pi 2/3.
    enabled:    bool  = True
    languages:  tuple = ("pt",)
    # A cada quantos frames rodar OCR (é pesado!)
    frame_interval: int = 20   # Aumentado de 15 para reduzir carga
    # Intervalo mínimo (s) para repetir o mesmo texto lido
    spam_interval: float = 10.0

OCR = OcrConfig()

# ---------------------------------------------------------------------------
# Display / Debug
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisplayConfig:
    # True para desenvolvimento/debug. No Pi headless, use False.
    # Consulte docs/pi2_deployment.md ou docs/pi3_deployment.md.
    show_window: bool  = True
    window_name: str   = "SIGA — Sistema Inteligente de Guiagem Assistiva"
    font_scale:  float = 0.55
    thickness:   int   = 2

DISPLAY = DisplayConfig()

# ---------------------------------------------------------------------------
# Classes de obstáculos que devem ser anunciados via TTS
# ---------------------------------------------------------------------------
# Dicionário de classes com prioridades (1 = Risco de vida/Fogo, 99 = Menor irrelevante)
OBSTACLE_PRIORITIES = {
    # Emergências críticas
    "Fogo": 1,
    "Fumaca": 2,
    "Buraco": 3,
    "Escada": 4,

    # Perigo iminente (Tráfego pesado)
    "Semaforo Vermelho": 5,
    "car": 6,
    "truck": 7,
    "bus": 8,
    "Veiculo": 9,
    "motorcycle": 10,
    
    # Navegação / Atenção Moderada
    "Piso Molhado": 11,
    "Semaforo Verde": 12,
    "Faixa de Pedestre": 13,
    "Saida": 14,
    
    # Obstáculos móveis ou inesperados
    "bicycle": 15,
    "person": 16,
    "Animal": 17,
    "dog": 18,
    
    # Obstáculos estáticos menores
    "Cone": 19,
    "Poste de Protecao": 20,
    "fire hydrant": 21,
    "bench": 22,
    "cat": 23,
}

OBSTACLE_CLASSES = frozenset(OBSTACLE_PRIORITIES.keys())

# Cores por fonte de detecção (BGR para OpenCV)
COLOR_URBANO  = (0,   255,   0)   # Verde
COLOR_ESQUINA = (0,   252, 254)   # Amarelo
COLOR_GERAL   = (255,   0,   0)   # Azul
