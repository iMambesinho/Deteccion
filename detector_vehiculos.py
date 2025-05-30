import cv2

def detectar_rostros():
    # Cargar el clasificador preentrenado de rostros
    cascada_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Capturar video desde la cámara
    captura = cv2.VideoCapture(0)
    
    while True:
        # Leer el frame
        ret, frame = captura.read()
        if not ret:
            break
        
        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros en la imagen
        rostros = cascada_rostro.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mostrar el frame con detecciones
        cv2.imshow("Detección Facial", frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_rostros()
