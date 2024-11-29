import os
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time

# Configuração do modelo e diretório para salvar fotos
model = YOLO(r"C:/Users/achal/runs/detect/train36/weights/best.pt")
output_dir = r"C:\Users\achal\Downloads\# Git\Treinamento YOLO Impressora 3D.v1.00\# Foto Time Lapse"

os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se não existir

cap = cv2.VideoCapture(1)

# Configurar a resolução máxima da webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Largura máxima suportada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Altura máxima suportada

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Nome e tamanho da janela
janela_nome = "Tela"
cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)  # Permite redimensionar
cv2.resizeWindow(janela_nome, 640, 360)  # Tamanho inicial da janela (exemplo: 640x360)

# Controle de tempo
last_capture_time = 0  # Armazena o tempo da última captura
last_photo_detect_time = 0  # Armazena o tempo de detecção da classe "Foto"
photo_cooldown = 30  # Tempo de espera após captura (em segundos)
detection_delay = 1  # Tempo de espera após detectar a classe "Foto" (em segundos)

while True:
    success, img = cap.read()

    if success:
        original_img = img.copy()  # Mantém uma cópia original sem alterações

        # Definir limite para margem esquerda (30% da largura da imagem)
        img_width = img.shape[1]  # Largura da imagem
        limite_x = img_width * 0.3  # 30% da largura

        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        # Processar os resultados
        for result in results:
            # Verificar se a classe "Foto" foi detectada
            for box, cls_id in zip(result.boxes.xywh.cpu(), result.boxes.cls.int().cpu()):
                if result.names[cls_id.item()] == "Foto":  # Verifica se a classe é "Foto"
                    # Obter a coordenada X do centro do bounding box
                    x_centro = box[0].item()

                    # Validar se está na margem esquerda (30%)
                    if x_centro <= limite_x:
                        # Verifica se passou o tempo de espera de 1 segundo após detectar "Foto"
                        current_time = time.time()
                        if current_time - last_photo_detect_time >= detection_delay:
                            # Verifica se passou o tempo de cooldown de 30 segundos
                            if current_time - last_capture_time >= photo_cooldown:
                                # Salvar a imagem original no diretório com qualidade máxima
                                img_name = f"{output_dir}/captura_{int(current_time)}.jpg"
                                cv2.imwrite(img_name, original_img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Qualidade máxima
                                print(f"Imagem salva: {img_name}")

                                # Atualiza o tempo da última captura e da última detecção de "Foto"
                                last_capture_time = current_time
                                last_photo_detect_time = current_time
                    else:
                        print(f"Classe 'Foto' detectada fora da margem permitida: X={x_centro:.2f}")

            # Visualizar os resultados na imagem
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    # Obter as boxes e IDs de rastreamento
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Plotar os rastros
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y centro
                        if len(track) > 30:  # Retém 30 rastros
                            track.pop(0)

                        # Desenhar as linhas de rastreamento
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        img = img.copy()  # Garante que a imagem é mutável
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except Exception as e:
                    print(f"Erro ao rastrear: {e}")

        # Redimensionar a janela ao tamanho especificado
        cv2.imshow(janela_nome, img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Desligando")
