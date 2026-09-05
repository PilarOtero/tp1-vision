import cv2
import numpy as np
import matplotlib.pyplot as plt

def achicar(img, max_dim = 1000):
    altura, ancho = img.shape[:2]
    escala = max_dim / max(altura, ancho)    
    # Si la imagen ya es mas chica que max_dim, no hacemos nada
    if escala < 1:
        img = cv2.resize(img, (int(ancho * escala), int(altura * escala)), interpolation=cv2.INTER_AREA)
    return img

def anms(keypoints, n_deseado):
    n = len(keypoints)
    if n <= n_deseado:
        return np.arange(n)

    puntos = np.array([kp.pt for kp in keypoints])
    respuestas = np.array([kp.response for kp in keypoints])

    #Definimos R_i como infinito para todos los elementos
    R = np.full(n, np.inf)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            #r_j > r_i
            if respuestas[j] > respuestas[i]:
                SD = (puntos[j, 0] - puntos[i, 0]) ** 2 + (puntos[j, 1] - puntos[i, 1]) ** 2
                if SD < R[i]:
                    R[i] = SD

    orden = np.argsort(R)[::-1]
    return orden[:n_deseado]

def aplicar_anms(keypoints, des, n_deseados):
    idx = anms(keypoints, n_deseados)
    keypoints_finales = [keypoints[i] for i in idx]
    descriptores_filtrados = des[idx]
    
    return keypoints_finales, descriptores_filtrados

def obtener_matches_cross_check(des_src, des_dst):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bf.match(des_src, des_dst)
    matches = sorted(matches, key=lambda m: m.distance)

    return matches

def obtener_matches_flann_lowe(des_src, des_dst, ratio=0.75):
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_knn = flann.knnMatch(des_src, des_dst, k=2)

    good_matches = []
    for match in matches_knn:
        if len(match) != 2:
            continue

        m, n = match

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda m: m.distance)

    return good_matches

def resumen_matches(matches, nombre):
    distancias = np.array([m.distance for m in matches])

    print(nombre)
    print(f"Cantidad: {len(matches)}")
    print(f"Distancia media: {distancias.mean():.2f}")
    print(f"Distancia mediana: {np.median(distancias):.2f}")
    print(f"Distancia mínima: {distancias.min():.2f}")
    print(f"Distancia máxima: {distancias.max():.2f}")
    print()

def mostrar_imagen_con_grilla(img, titulo="", paso=50, figsize=(12, 10)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(titulo)

    ax.set_xticks(np.arange(0, w + 1, paso))
    ax.set_yticks(np.arange(0, h + 1, paso))
    ax.grid(color="yellow", linestyle="-", linewidth=0.5, alpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.show()

def dlt(ori, dst):

    # Construct matrix A and vector b
    A = []
    b = []
    for i in range(4):
        x, y = ori[i]
        x_prima, y_prima = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima])
        b.append(x_prima)
        b.append(y_prima)

    A = np.array(A)
    b = np.array(b)

    # resolvemos el sistema de ecuaciones A * h = b
    # el sistema es de 8x8, por lo que podemos resolverlo si A es inversible

    # resuelve el sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)

    # agrega el elemento h_33
    H = np.hstack([H, [1]])

    # reorganiza H para formar la matrix en 3x3 to form the 3x3 homography matrix
    H = H.reshape(3, 3)

    return H