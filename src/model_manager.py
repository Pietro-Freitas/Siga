"""
model_manager.py — Orquestrador de Modelos YOLO
================================================
Centraliza o carregamento, o health-check e a estratégia de inferência
dos modelos YOLO, evitando redundância e sobrecarga de VRAM/RAM.

Estratégia de orquestração (economia de ~60–70% de CPU no Pi):
  • yolov8n (geral)   → roda em TODOS os frames (base leve).
  • modelo_urbano     → roda a cada N frames (MODEL.urbano_frame_interval).
  • modelo_esquina    → roda a cada M frames (MODEL.esquina_frame_interval).
  • OCR (EasyOCR)    → roda a cada K frames (OCR.frame_interval).

NOTA DE PRODUÇÃO — EXPORTAÇÃO ONNX:
  Para ~2× de speedup no Raspberry Pi, exporte os modelos para ONNX:
      from ultralytics import YOLO
      model = YOLO('modelos/best.pt')
      model.export(format='onnx', imgsz=320, half=False)
  Depois troque a extensão em config.py:
      urbano: Path = MODEL_DIR / "best.onnx"
  O código abaixo detecta automaticamente modelos .onnx e os carrega
  sem necessidade de PyTorch (muito mais leve em RAM).
  Consulte docs/pi2_deployment.md ou docs/pi3_deployment.md para instruções detalhadas.

NOTA ATUAL — 3 MODELOS SIMULTÂNEOS:
  Atualmente carregamos geral + urbano + esquina para testes.
  Para deploy final, use apenas UM modelo (provavelmente o urbano/best.pt).
  Consulte docs/pi2_deployment.md ou docs/pi3_deployment.md para a configuração recomendada.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import COLOR_ESQUINA, COLOR_GERAL, COLOR_URBANO, MODEL, OCR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports condicionais (lazy) — não travar se libs estiverem ausentes
# ---------------------------------------------------------------------------
_YOLO_OK = False
_OCR_OK  = False

try:
    from ultralytics import YOLO

    # -----------------------------------------------------------------------
    # NOTA DE SEGURANÇA — torch.load com weights_only=False
    # -----------------------------------------------------------------------
    # O PyTorch 2.6+ exige weights_only=True por padrão para prevenir
    # execução de código malicioso via pickle em arquivos .pt.
    #
    # O Ultralytics YOLO usa classes customizadas nos checkpoints que não
    # são compatíveis com weights_only=True sem um allowlist extenso.
    #
    # A solução abaixo restaura o comportamento pre-2.6 (weights_only=False).
    # Isso é SEGURO no nosso contexto porque:
    #   1. Só carregamos modelos que NÓS treinamos.
    #   2. Os arquivos .pt nunca vêm de fontes não confiáveis.
    #
    # ⚠️  NUNCA carregue um arquivo .pt de origem desconhecida com este patch.
    # Para produção com modelos de terceiros, exporte para ONNX (mais seguro).
    # -----------------------------------------------------------------------
    import functools
    import torch
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _safe_load_for_known_models(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _safe_load_for_known_models
    _YOLO_OK = True
except ImportError:
    logger.error("ultralytics não instalada. Visão desabilitada.")

try:
    import easyocr
    _OCR_OK = True
except ImportError:
    logger.warning("easyocr não instalada. OCR desabilitado.")


# ---------------------------------------------------------------------------
# DTOs de resultado
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Representa uma única detecção YOLO normalizada."""
    x1:    int
    y1:    int
    x2:    int
    y2:    int
    name:  str
    conf:  float
    color: tuple   # BGR para cv2


@dataclass
class InferenceResult:
    """Resultado completo de um ciclo de inferência sobre um frame."""
    detections: list[Detection] = field(default_factory=list)
    ocr_texts:  list[str]       = field(default_factory=list)
    frame_id:   int             = 0
    latency_ms: float           = 0.0


# ---------------------------------------------------------------------------
# Gerenciador principal
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Gerencia o ciclo de vida e a orquestração de todos os modelos de IA.

    Carrega cada modelo uma única vez e controla quando cada um é acionado,
    de acordo com os intervalos de frames configurados em `config.py`.
    """

    def __init__(self) -> None:
        self._modelo_geral:   Optional[object] = None
        self._modelo_urbano:  Optional[object] = None
        self._modelo_esquina: Optional[object] = None
        self._ocr_reader:     Optional[object] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Carregamento & health-check
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Carrega todos os modelos disponíveis em disco.

        Returns
        -------
        True se o modelo base (geral) foi carregado com sucesso.
        Modelos opcionais são carregados com best-effort.
        """
        if not _YOLO_OK:
            logger.error("ultralytics ausente — nenhum modelo carregado.")
            return False

        # Modelo base — obrigatório
        ok = self._load_model("geral", MODEL.geral)
        if not ok:
            logger.error("Modelo geral (obrigatório) não pôde ser carregado.")
            return False

        # Modelos especializados — opcionais (best-effort)
        self._load_model("urbano",  MODEL.urbano)
        self._load_model("esquina", MODEL.esquina)

        # OCR — opcional e pesado; só carrega se lib disponível E habilitado
        if _OCR_OK and OCR.enabled:
            self._load_ocr()
        elif not OCR.enabled:
            logger.info("OCR desabilitado via config (OCR.enabled=False).")

        self._loaded = True
        logger.info(
            "ModelManager pronto. OCR=%s, Urbano=%s, Esquina=%s",
            self._ocr_reader is not None,
            self._modelo_urbano is not None,
            self._modelo_esquina is not None,
        )
        return True

    def _load_model(self, nome: str, path: Path) -> bool:
        """Carrega um modelo YOLO individual."""
        if not path.exists():
            logger.warning("Modelo '%s' não encontrado: %s", nome, path)
            return False
        try:
            model = YOLO(str(path))
            setattr(self, f"_modelo_{nome}", model)
            logger.info("Modelo '%s' carregado: %s", nome, path.name)
            return True
        except Exception:
            logger.exception("Falha ao carregar modelo '%s'.", nome)
            return False

    def _load_ocr(self) -> None:
        """Carrega o EasyOCR. Pesado (~300-500MB RAM)."""
        try:
            self._ocr_reader = easyocr.Reader(
                list(OCR.languages), gpu=False, verbose=False
            )
            logger.info("EasyOCR carregado (idiomas: %s).", OCR.languages)
        except Exception:
            logger.exception("Falha ao carregar EasyOCR.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._modelo_geral is not None

    # ------------------------------------------------------------------
    # Inferência orquestrada
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray, frame_id: int) -> InferenceResult:
        """
        Executa inferência orquestrada no frame fornecido.

        O modelo geral roda sempre. Os modelos especializados obedecem
        seus respectivos intervalos de frame. O OCR obedece seu intervalo.

        Parameters
        ----------
        frame:    Frame BGR capturado pela câmera.
        frame_id: Contador incremental de frames.

        Returns
        -------
        InferenceResult com todas as detecções e textos do ciclo.
        """
        if not self.is_ready:
            return InferenceResult(frame_id=frame_id)

        t0 = time.perf_counter()
        result = InferenceResult(frame_id=frame_id)

        # ---- 1. Modelo geral (sempre) ----
        result.detections.extend(
            self._run_yolo(
                self._modelo_geral, frame,
                MODEL.conf_geral, COLOR_GERAL,
                ignored_classes=MODEL.ignored_classes_geral,
            )
        )

        # ---- 2. Modelo urbano (a cada N frames) ----
        if (self._modelo_urbano is not None
                and frame_id % MODEL.urbano_frame_interval == 0):
            result.detections.extend(
                self._run_yolo(
                    self._modelo_urbano, frame,
                    MODEL.conf_urbano, COLOR_URBANO,
                )
            )

        # ---- 3. Modelo esquina (a cada M frames) ----
        if (self._modelo_esquina is not None
                and frame_id % MODEL.esquina_frame_interval == 0):
            result.detections.extend(
                self._run_yolo(
                    self._modelo_esquina, frame,
                    MODEL.conf_esquina, COLOR_ESQUINA,
                )
            )

        # ---- 4. OCR (a cada K frames, se disponível e habilitado) ----
        if (self._ocr_reader is not None
                and OCR.enabled
                and frame_id % OCR.frame_interval == 0):
            result.ocr_texts = self._run_ocr(frame)

        result.latency_ms = (time.perf_counter() - t0) * 1000

        # Log apenas se latência significativa (reduz I/O de logging)
        if result.latency_ms > 100 or result.detections:
            logger.debug(
                "Frame %d — %d det, %d OCR, %.0fms",
                frame_id, len(result.detections),
                len(result.ocr_texts), result.latency_ms,
            )
        return result

    # ------------------------------------------------------------------
    # Helpers de inferência
    # ------------------------------------------------------------------

    def _run_yolo(
        self,
        model,
        frame: np.ndarray,
        conf: float,
        color: tuple,
        ignored_classes: tuple = (),
    ) -> list[Detection]:
        """Executa um modelo YOLO e retorna lista de Detection."""
        try:
            # imgsz reduz o frame internamente antes da inferência.
            # Usar valor menor = menos processamento. 320 é ideal para Pi 2.
            res = model(frame, conf=conf, imgsz=MODEL.imgsz, verbose=False)[0]
        except Exception:
            logger.exception("Erro de inferência YOLO.")
            return []

        if res.boxes is None or len(res.boxes) == 0:
            return []

        # Transferência GPU→CPU em operação única (evita 3 chamadas .cpu())
        boxes_data = res.boxes.data.cpu().numpy()

        detections: list[Detection] = []
        names = model.names

        for row in boxes_data:
            cls_id = int(row[5])
            if cls_id in ignored_classes:
                continue
            detections.append(Detection(
                x1=int(row[0]), y1=int(row[1]),
                x2=int(row[2]), y2=int(row[3]),
                name=names[cls_id],
                conf=float(row[4]),
                color=color,
            ))
        return detections

    def _run_ocr(self, frame: np.ndarray) -> list[str]:
        """
        Executa OCR e retorna lista de strings detectadas.

        O EasyOCR lê textos visíveis no frame (placas, letreiros, etc.)
        e os retorna como strings para serem anunciados via TTS.

        NOTA: Muito pesado em CPU (~300-500ms por frame no Pi 2).
        Considere desabilitar via OCR.enabled=False para feiras.
        """
        try:
            return self._ocr_reader.readtext(frame, detail=0, paragraph=True)
        except Exception:
            logger.exception("Erro no OCR.")
            return []
