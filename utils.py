import cv2
import numpy as np

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