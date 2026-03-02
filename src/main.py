import easyocr
from ultralytics import YOLO
import cv2
import pyttsx3
import sounddevice as sd
import json
import vosk
import threading

procurando = False
objetivo = ''
encontrado = False
ultimo_texto = ''

#MODELOS
reader = easyocr.Reader(['pt'])
modelo_urbano = YOLO('runs/detect/train3/weights/best.pt')
modelo_geral = YOLO('yolov8m.pt')
modelo_esquina = YOLO('esquina.pt')
modelo_comando = vosk.Model('vosk-model-small-pt-0.3')
rec = vosk.KaldiRecognizer(modelo_comando, 16000)

#VOZ
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)

def falar(texto):
    engine.say(texto)
    engine.runAndWait()

#CALLBACK DE ÁUDIO
def callback(indata, frames, time, status):
    global procurando, objetivo, encontrado

    if rec.AcceptWaveform(bytes(indata)):
        result = json.loads(rec.Result())
        text = result.get('text', '').strip()
        if not text:
            return

        print("Comando:", text)

        if text == 'encontrar':
            falar('O que deseja localizar?')
            procurando = True
            encontrado = False
            return

        if text == 'parar':
            procurando = False
            objetivo = ''
            encontrado = False
            falar('Busca cancelada')
            return

        if procurando:
            objetivo = text
            encontrado = False
            falar(f'Procurando {objetivo}')

#ÁUDIO
def iniciar_audio():
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        sd.sleep(999999)

threading.Thread(target=iniciar_audio, daemon=True).start()

#CÂMERA
cap = cv2.VideoCapture(0)
print(modelo_urbano.names)
print(modelo_geral.names)

while True:
    acesso, frame = cap.read()
    if not acesso:
        break

    #OCR
    res_texto = reader.readtext(frame, detail=0, paragraph=True)

    #YOLO urbano primeiro
    res_urbano = modelo_urbano(frame, conf=0.45, verbose=False)[0]
    res_esquina = modelo_esquina(frame, conf=0.45, verbose=False)[0]

    if len(res_urbano.boxes) > 0:
        caixas = res_urbano.boxes.xyxy.cpu().numpy()
        confs = res_urbano.boxes.conf.cpu().numpy()
        clss = res_urbano.boxes.cls.cpu().numpy().astype(int)
        nomes = modelo_urbano.names
        cor = (0, 255, 0)
        
    if len(res_esquina.boxes) > 0:
        caixas = res_esquina.boxes.xyxy.cpu().numpy()
        confs = res_esquina.boxes.conf.cpu().numpy()
        clss = res_esquina.boxes.cls.cpu().numpy().astype(int)
        nomes = modelo_esquina.names
        cor = (254, 252, 0)
    else:
        res_geral = modelo_geral(frame, conf=0.45, verbose=False)[0]
        caixas = res_geral.boxes.xyxy.cpu().numpy()
        confs = res_geral.boxes.conf.cpu().numpy()
        clss = res_geral.boxes.cls.cpu().numpy().astype(int)
        nomes = modelo_geral.names
        cor = (255, 0, 0)

    #DETECÇÕES
    for caixa, conf, cls in zip(caixas, confs, clss):
        if cls in [12, 24]:
            continue

        x1, y1, x2, y2 = map(int, caixa)
        nome = nomes[cls]

        if objetivo == 'pessoa' and nome == 'person' and not encontrado:
            falar('Pessoa encontrada')
            encontrado = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(
            frame,
            f'{nome} {conf:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor,
            2
        )

    #OCR COM CONTROLE
    for fala in res_texto:
        if fala != ultimo_texto:
            print("Texto:", fala)
            falar(fala)
            ultimo_texto = fala

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
