# Programa para aplicar distintos filtros a una imagen usando OpenCV y matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_img(img):
    # Preguntar si el usuario quiere guardar la imagen
    # Si sí, pedir nombre de archivo y guardar
    # Si no, solo imprimir mensaje
    save_select = input("Do you wish to save the image?\n1. Yes\n2. No\n> ")

    if save_select == "1":
        filename = input("Enter a filename (with .jpg or .png extension):\n> ")
        success = cv2.imwrite(filename, img)
        if success:
            print(f"✅ Image saved as '{filename}'")
        else:
            print("❌ Failed to save image. Check path or filename.")
    elif save_select == "2":
        print("🚫 Image not saved.")
    else:
        print("⚠️ Invalid option. Image not saved.")
    return

def apply_sepia_filter(img):
    # Aplica filtro sepia a la imagen
    # Normalizar la imagen para aplicar matriz sepia
    img = np.array(img, dtype=np.float32) / 255.0

    # Matriz sepia en formato BGR
    sepia_bgr = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393]
    ])

    # Aplicar transformación sepia
    sepia = cv2.transform(img, sepia_bgr)

    # Limitar valores [0,1] y regresar a 8 bits
    sepia = np.clip(sepia, 0, 1)
    sepia = (sepia * 255).astype(np.uint8)

    # Convertir a RGB para mostrar y guardar
    sepia_rgb = cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB)

    plt.imshow(sepia_rgb)
    plt.title("Filtro SEPIA cálido")
    plt.axis("off")
    plt.show()
    # Preguntar si se quiere guardar
    save_img(sepia_rgb)
    return

def apply_negative_filter(img):
    # Aplica filtro negativo a la imagen
    # Aplicar inversión de colores
    img_neg = cv2.bitwise_not(img)
    plt.imshow(cv2.cvtColor(img_neg, cv2.COLOR_BGR2RGB))
    plt.title("Filtro Negativo")
    plt.axis("off")
    plt.show()
    save_img(img_neg)

def apply_popart_filter(img):
    # Filtro estilo pop-art con 4 combinaciones de color

    # --- 1. Cargar la imagen ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (400, 400))  # tamaño uniforme

    # --- 2. Convertir a escala de grises ---
    gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    # --- 3. Ecualizar contraste y normalizar ---
    gray_eq = cv2.equalizeHist(gray)
    gray_norm = gray_eq / 255.0

    # Función para aplicar combinación de color específica
    def apply_pop_color(gray_img, color):
        g = np.clip(gray_img * 255, 0, 255).astype(np.uint8)
        color_map = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
        
        if color == 'pink':
            color_map[:,:,0] = g        # R
            color_map[:,:,1] = (255 - g)//2  # G
            color_map[:,:,2] = (255 - g)     # B
        elif color == 'green':
            color_map[:,:,0] = (255 - g)//2
            color_map[:,:,1] = g
            color_map[:,:,2] = (255 - g)
        elif color == 'blue':
            color_map[:,:,0] = (255 - g)
            color_map[:,:,1] = (255 - g)//2
            color_map[:,:,2] = g
        elif color == 'yellow':
            color_map[:,:,0] = g
            color_map[:,:,1] = g
            color_map[:,:,2] = (255 - g)//4
        elif color == 'orange':
            color_map[:,:,0] = g
            color_map[:,:,1] = (g*0.5 + 100).clip(0,255)
            color_map[:,:,2] = (255 - g)//2
        elif color == 'purple':
            color_map[:,:,0] = (g*0.6 + 50).clip(0,255)
            color_map[:,:,1] = (255 - g)//3
            color_map[:,:,2] = g
        else:
            color_map[:,:,0] = g
            color_map[:,:,1] = g
            color_map[:,:,2] = g
        return color_map

    # --- 5. Generar 4 versiones con diferentes tonos ---
    img1 = apply_pop_color(gray_norm, 'pink')
    img2 = apply_pop_color(gray_norm, 'green')
    img3 = apply_pop_color(gray_norm, 'yellow')
    img4 = apply_pop_color(gray_norm, 'blue')

    # Unir imágenes en cuadrícula 2x2 estilo Warhol
    top = np.hstack((img1, img2))
    bottom = np.hstack((img3, img4))
    pop_art = np.vstack((top, bottom))

    # --- 7. Mostrar resultado ---
    plt.figure(figsize=(8,8))
    plt.imshow(pop_art)
    plt.title("Filtro POP-ART 🎨")
    plt.axis("off")
    plt.show()
    save_img(pop_art)

def apply_bw_filter(img):
    # Convertir imagen a blanco y negro
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(bw_image, cmap="gray")
    plt.title("Filtro B&W")
    plt.axis("off")
    plt.show()
    save_img(bw_image)

def apply_warm_filter(img):
    # Aumentar canal rojo y disminuir azul para efecto cálido

    # --- 2. Aplicar efecto cálido en BGR ---
    warm = img.astype(np.float32)

    # En BGR: canal 0 = azul, canal 2 = rojo
    warm[:,:,0] = np.clip(warm[:,:,0] * 0.9, 0, 255)   # Azul ↓
    warm[:,:,2] = np.clip(warm[:,:,2] * 1.2, 0, 255)   # Rojo ↑

    warm = warm.astype(np.uint8)

    # --- 3. Convertir a RGB solo para mostrar ---
    warm_rgb = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(warm_rgb)
    plt.title("Filtro Cálido ☀️")
    plt.axis("off")
    plt.show()
    save_img(warm_rgb)

def apply_cold_filter(img):
    # Aumentar azul y reducir rojo para efecto frío

    # --- 2. Aplicar efecto frío en BGR ---
    cool = img.astype(np.float32)

    # En BGR: canal 0 = azul, canal 2 = rojo
    cool[:,:,0] = np.clip(cool[:,:,0] * 1.2, 0, 255)   # Azul ↑
    cool[:,:,2] = np.clip(cool[:,:,2] * 0.9, 0, 255)   # Rojo ↓

    cool = cool.astype(np.uint8)

    # --- 3. Convertir a RGB solo para mostrar ---
    cool_rgb = cv2.cvtColor(cool, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- 4. Mostrar comparación ---
    plt.imshow(cool_rgb)
    plt.title("Filtro Frío ❄️")
    plt.axis("off")
    plt.show()
    save_img(cool_rgb)

def apply_drawing_filter(img):
    # Convertir imagen a tipo dibujo con efecto lápiz
    # --- 4️⃣ Convertir a escala de grises ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 5️⃣ Invertir los tonos ---
    inv = 255 - gray

    # --- 6️⃣ Aplicar desenfoque para suavizar la inversión ---
    blur = cv2.GaussianBlur(inv, (25, 25), 0)

    # --- 7️⃣ Combinar la imagen original con el desenfoque invertido ---
    # Esta división produce un efecto tipo lápiz
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # --- 8️⃣ Mostrar resultado ---
    plt.imshow(sketch, cmap='gray')
    plt.title("Filtro Sketch ✏️ (Dibujo a lápiz)")
    plt.axis("off")
    plt.show()

    # --- 9️⃣ (Opcional) Comparar con la imagen original ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    plt.imshow(sketch, cmap='gray')
    plt.title("Sketch ✏️")
    plt.axis("off")
    plt.show()
    save_img(sketch)


# Pedir al usuario la ruta de la imagen
path = input("\n\nPlease write the route of the photo you wish to edit: ")
img = cv2.imread(path)

# Menú interactivo para aplicar filtros a la imagen
while True:
    choice = input(
        "What filter should we apply?\n"
        "1. Sepia\n"
        "2. Negative\n"
        "3. Pop-Art\n"
        "4. B&W\n"
        "5. Warm Filter\n"
        "6. Cold Filter\n"
        "7. Drawing\n"
        "8. Exit\n"
        "Your choice: "
    )

    if choice == "1":
        apply_sepia_filter(img)
    elif choice == "2":
        apply_negative_filter(img)
    elif choice == "3":
        apply_popart_filter(img)
    elif choice == "4":
        apply_bw_filter(img)
    elif choice == "5":
        apply_warm_filter(img)
    elif choice == "6":
        apply_cold_filter(img)
    elif choice == "7":
        apply_drawing_filter(img)
    elif choice == "8":
        break
    else:
        print("❌ Invalid choice. Please try again.")
# Mensaje de despedida al salir del programa
print("Thanks for using our program!!")
