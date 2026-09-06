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

def obtener_matches(des_src, des_dst, crossCheck):
    # Cross-checking: un match (i,j) solo es valido si j es el mejor vecino de i e i es el mejor vecino de j
    # La norma 2 calcula la distancia euclidea entre los descriptores, la cual es EXACTA
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = crossCheck)
    matches = bf.match(des_src, des_dst)
    matches = sorted(matches, key = lambda m: m.distance)

    return matches

def obtener_matches_flann_lowe(des_src, des_dst, ratio = 0.75):
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_knn = flann.knnMatch(des_src, des_dst, k = 2)

    good_matches = []
    for match in matches_knn:
        if len(match) != 2:
            continue

        m, n = match

        # m.distance / n.distance < ratio
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key = lambda m: m.distance)
    return good_matches

def resumen_matches(matches, nombre):
    distancias = np.array([m.distance for m in matches])

    return {
        'metodo': nombre,
        'cantidad': len(matches),
        'distancia_media': np.mean(distancias),
        'distancia_mediana': np.median(distancias),
        'distancia_minima': np.min(distancias),
        'distancia_maxima': np.max(distancias)
    }

def obtener_matches_combinado(desc_src, desc_dst, ratio = 0.75):
    # Obtenemos los matches con cross-check ya que era el metodo que mas hallaba
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)

    # Calculamos los matches con FLANN + Lowe y cross-check (para obtener una distancia exacta)
    knn_src_dest = bf.knnMatch(desc_src, desc_dst, k = 2)
    knn_dst_src = bf.knnMatch(desc_dst, desc_src, k = 2)

    # De source a destino, guardo el mejor vecino solo si pasa el ratio de Lowe
    buenos_vecinos_src_dst = {}
    for m, n in knn_src_dest:
        if m.distance < ratio * n.distance:
            buenos_vecinos_src_dst[m.queryIdx] = m

    # De destino a source, guardo el mejor vecino solo si pasa el ratio de Lowe
    buenos_vecinos_dst_src = {}
    for m, n in knn_dst_src:
        if m.distance < ratio * n.distance:
            buenos_vecinos_dst_src[m.queryIdx] = m.trainIdx

    # Cross-check: un match (i,j) 
    matches =[]
    for i, m in buenos_vecinos_src_dst.items():
        j = m.trainIdx
        if buenos_vecinos_dst_src.get(j) == i:
            matches.append(m)

    matches = sorted(matches, key = lambda m: m.distance)
    return matches

def mostrar_cambios(img2, keypoints2, img1, keypoints1, matches_2_1, inliers_mask = None):
    if inliers_mask is None:
        img_2_1 = cv2.drawMatches(
            img2, keypoints2,
            img1, keypoints1,
            matches_2_1, None,
            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return img_2_1

    inliers_mask = inliers_mask.astype(int)
    img_2_1 = cv2.drawMatches(img2, [], img1, [], [], None)

    cv2.drawMatches(
        img2, keypoints2,
        img1, keypoints1,
        matches_2_1,
        outImg = img_2_1,
        matchesMask = inliers_mask.tolist(),
        matchColor = (0, 255, 0),
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    )
    cv2.drawMatches(
        img2, keypoints2,
        img1, keypoints1,
        matches_2_1,
        outImg = img_2_1,
        matchesMask = (1 - inliers_mask).tolist(),
        matchColor = (0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    )
    return img_2_1


def mostrar_imagen_con_grilla(img, titulo = "", paso = 50, figsize = (12, 10)):
    if isinstance(img, (list, tuple)):
        imagenes = img
        titulos = titulo
        if isinstance(titulos, str):
            titulos = [titulos] * len(imagenes)

        _, axes = plt.subplots(1, len(imagenes), figsize=figsize)
        if len(imagenes) == 1:
            axes = [axes]

        for ax, imagen, titulo_i in zip(axes, imagenes, titulos):
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            ax.imshow(img_rgb)
            ax.set_title(titulo_i)
            ax.set_xticks(np.arange(0, w + 1, paso))
            ax.set_yticks(np.arange(0, h + 1, paso))
            ax.grid(color="yellow", linestyle="-", linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        plt.tight_layout()
        plt.show()
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(titulo)

    ax.set_xticks(np.arange(0, w + 1, paso))
    ax.set_yticks(np.arange(0, h + 1, paso))
    ax.grid(color = "yellow", linestyle="-", linewidth = 0.5, alpha = 0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.show()

def dlt(ori, dst):
    A = []
    b = []

    # Construimos el sistema de ecuaciones lineales a partir de las correspondencias
    for i in range(4):
        x, y = ori[i]
        x_prima, y_prima = dst[i]
        # Cada correspondencia genera dos ecuaciones lineales
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima])
        b.append(x_prima)
        b.append(y_prima)

    A = np.array(A)
    b = np.array(b)
    # Resolvemos el sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)
    # Agregamos el elemento h_33
    H = np.hstack([H, [1]])

    H = H.reshape(3, 3)
    return H

def computar_distancias_reproyeccion(H, pts1, pts2):
    # Convertir pts1 a coordenadas homogéneas
    pts1_hom = np.hstack([pts1, np.ones((len(pts1), 1))])
    
    # Proyecta pts1 usando H
    pts1_proy = np.dot(H, pts1_hom.T).T
    pts1_proy = pts1_proy[:, :2] / pts1_proy[:, 2].reshape(-1, 1) #normalizar
    
    dist = np.linalg.norm(pts1_proy - pts2, axis = 1)
    return dist 
    
def error_reproyeccion(H, pts1, pts2):
    dist = computar_distancias_reproyeccion(H, pts1, pts2)
    
    rmse = np.sqrt(np.mean(dist**2))
    return rmse

def homografia_svd(pts1, pts2):
    A = []
    for (x, y), (x_prima, y_prima) in zip(pts1, pts2):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima, x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima, y_prima])
    A = np.array(A)

    # SVD (agarramos el vector correspondiente al menor valor singular)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2,2]

def puntos_con_matches(keypoints_src, keypoints_dst, matches):
    pts_src = np.float32([keypoints_src[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([keypoints_dst[m.trainIdx].pt for m in matches])
    return pts_src, pts_dst

def ransac(pts1, pts2, T = 1000, umbral = 5.0, seed = 42):
    rng = np.random.default_rng(seed)
    n = len(pts1)

    mejores_inliers = []
    mejor_cantidad = -1

    for _ in range(T):
        # 1. Seleccionar 4 pares de correspondencia aleatorias
        idx = rng.choice(n, size = 4, replace = False)

        # 2. Calcular la homografía H usando DLT
        try:
            H = dlt(pts1[idx], pts2[idx])
        #  Si la matriz A es singular, saltamos esta iteración
        except np.linalg.LinAlgError:
            continue  

        # 3. Determinar correspondencias inliers tal que la distancia de reproyección sea menor que el umbral
        distancias = computar_distancias_reproyeccion(H, pts1, pts2)
        inliers = distancias < umbral
        cantidad = inliers.sum()

        # 4. Recordar el conjunto de inliers mas grande
        if cantidad > mejor_cantidad:
            mejor_cantidad = cantidad
            mejores_inliers = inliers

    # 5. Recalcular H usando cuadrdaos minimos utilizando todos los inliers
    H_final = homografia_svd(pts1[mejores_inliers], pts2[mejores_inliers])
    error_final = error_reproyeccion(H_final, pts1[mejores_inliers], pts2[mejores_inliers])

    return H_final, error_final, mejores_inliers

def homograficas_a_euclideas(H, puntos):
    puntos = np.asarray(puntos, dtype = np.float64)
    puntos_homograficos = np.hstack([puntos, np.ones((len(puntos), 1))])
    transformados = (H @ puntos_homograficos.T).T

    return transformados[:,:2] / transformados[:,2:3]
