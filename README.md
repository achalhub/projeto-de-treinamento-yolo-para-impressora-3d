# Guia de Configuração: YOLO para Detecção em Impressoras 3D

Este guia descreve as alterações necessárias para configurar os arquivos do projeto em diferentes máquinas.

---

## 1. Arquivo `train_impressora3d_v8.py`

Este arquivo é usado para treinar o modelo YOLO.

### Variáveis a modificar:
- **Caminho dos dados para treinamento**:
  - Atualize o caminho do arquivo `.yaml` que contém as configurações do dataset.
    ```python
    model.train(data='caminho/para/impressora.yaml', epochs=50)
    ```
- **Configuração da GPU/CPU**:
  - Certifique-se de que o treinamento está configurado para usar GPU, caso disponível:
    ```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ```
- **Número de épocas**:
  - Ajuste o parâmetro `epochs` para adequar à capacidade de hardware e ao tempo disponível:
    ```python
    model.train(epochs=50)
    ```

---

## 2. Arquivo `impressora.yaml`

Este arquivo configura os caminhos e as classes do dataset.

### Variáveis a modificar:
- **Caminhos dos dados**:
  - Atualize os caminhos para as pastas de imagens de treinamento e validação:
    ```yaml
    train: caminho/para/dados/treino
    val: caminho/para/dados/validacao
    ```
- **Classes de objetos**:
  - Defina as classes que o modelo precisa detectar:
    ```yaml
    names:
      0: "Classe1"
      1: "Classe2"
    ```

---

## 3. Arquivo `detectar_usando_webcam.py`

Este script é responsável por usar o modelo treinado para detectar objetos em tempo real.

### Variáveis a modificar:
- **Caminho do modelo YOLO treinado**:
  - Atualize o caminho para o arquivo `.pt` gerado durante o treinamento:
    ```python
    model = YOLO("caminho/para/best.pt")
    ```
- **Dispositivo de captura de vídeo**:
  - Configure o índice correto da câmera no sistema:
    ```python
    cap = cv2.VideoCapture(0)  # Use 0, 1 ou outro índice dependendo do hardware
    ```
- **Resolução da captura de vídeo**:
  - Ajuste a resolução da câmera para compatibilidade com o dispositivo:
    ```python
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ```

---

## Checklist Geral

### 1. Instalar Dependências
- Certifique-se de que todas as bibliotecas requeridas estão instaladas:
  ```bash
  pip install -r requirements.txt
