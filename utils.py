import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def achicar(img:np.ndarray, max_dim:int = 1000) -> np.ndarray:
    """
    Achica imagenes de más de 1000 pixeles para facilitar su procesamiento. Aquellas con menos de esa cantidad, mantienen su tamaño original
    
        Parámetros:
            img(np.ndarray): imagen a achicar
            max_dim(int): máxima cantidad de pixeles que debe tener la imagen final

        Retorna:
            img(np.ndarray): la imagen achicada u original en caso de no haber sido necesario reducir su dimensión
    """
    altura, ancho = img.shape[:2]
    escala = max_dim / max(altura, ancho)
    # Si la imagen ya es mas chica que max_dim, no hacemos nada
    if escala < 1:
        img = cv2.resize(img, (int(ancho * escala), int(altura * escala)), interpolation = cv2.INTER_AREA)
    return img

def factor_escala(img:np.ndarray, max_dim:int = 1000):
    """
    Calcula el minimo entre 1 y el cociente entre la máxima dimensión aceptada y el máximo entre el ancho y alto de la imagen
    
        Parámetros:
            img(np.ndarray): imágen a analizar
            max_dim(int): máxima dimensión (cantidad de pixeles) aceptada

        Retorna:
            (int): mínimo entre 1 y el cociente entre max_dim y el máximo entre el ancho y alto de la imagen
    """
    altura, ancho = img.shape[:2]
    return min(1.0, max_dim / max(altura, ancho))

def escalar_homografia(H:np.ndarray, s_src:float, s_dst:float):
    """
    Reescala una homografía calculada sobre imágenes achicadas, para poder aplicarla directamente
    sobre las imágenes originales. Como escalar una imagen equivale a multiplicar sus coordenadas por un factor, 
    hay que "deshacer" ese escalado antes y después de H

        Parámetros:
            H(np.ndarray): homografía (3x3) estimada sobre las imágenes achicadas
            s_src(float): factor de escala aplicado a la imagen de origen (achicar_size / tamaño_original)
            s_dst(float): factor de escala aplicado a la imagen de destino

        Retorna:
            np.ndarray: homografía equivalente (3x3), válida para las imágenes a resolución original
    """
    # Las homografias fueron estimadas sobre imagenes achicadas por s_src y s_dst.
    # Para aplicarlas sobre las imagenes originales hay que deshacer ese escalado:
    # x_full = S_dst^-1 @ H_scaled @ S_src @ x_full, con S = diag(s, s, 1)
    S_src = np.diag([s_src, s_src, 1.0])
    S_dst = np.diag([s_dst, s_dst, 1.0])
    return np.linalg.inv(S_dst) @ H @ S_src

# Para esta función se utilizó como guía el pseudocodigo provisto
def anms(keypoints:cv2.KeyPoint, n_deseado:int) -> np.ndarray:
    """
    Algoritmo de Supresión No-Máxima Adaptativa que permite reducir la cantidad total de keypoints hallados, manteniendo solo aquellos 'bien' 
    distribuidos en la imagen
        
        Parámetros:
            keypoints(cv2.Keypoint): características visuales de la imagen objetidas con SIFT
            n_deseado(int): cantidad deseada de características visuales (keypoints) a tener

        Retorna:
            orden(np.ndarray de int): array de los mejores n_deseados keypoints
    """
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
            
            if respuestas[j] > respuestas[i]:
                SD = (puntos[j, 0] - puntos[i, 0]) ** 2 + (puntos[j, 1] - puntos[i, 1]) ** 2
                if SD < R[i]:
                    R[i] = SD

    # Ordenamos la lista de puntos para que el indice de los mejores se encuentren al inicio de la lista
    orden = np.argsort(R)[::-1]

    # Nos quedamos con los indices en los que se encuentran los mejores n_deseados keypoints
    return orden[:n_deseado]

def aplicar_anms(keypoints:cv2.KeyPoint, des:np.ndarray, n_deseados:int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Aplica la función anms implementada a los keypoints hallados
        
        Parámetros:
            keypoints(cv2.Keypoint): características visuales de la imagen objetidas con SIFT
            des(np.ndarray): array de los descriptores de los keypoints de la forma (cantidad_de_keypoints, 128) que dice cómo es numéricamente la 
            zona de la imagen alrededor de ese punto
            n_deseado(int): cantidad deseada de características visuales (keypoints) a tener

        Retorna:
            keypoints_finales(cv2.Keypoint): características visuales de la imagen que "sobrevivieron" a la aplicación de ANMS
            descriptores_filtrados(np.ndarray): array de los descriptores de los keypoints_finales
    """
    # ANMS devuelve los indices de los keypoints
    idx = anms(keypoints, n_deseados)
    # Alineamos keypoint y descriptor 
    keypoints_finales = [keypoints[i] for i in idx]
    descriptores_filtrados = des[idx]
    
    return keypoints_finales, descriptores_filtrados

def obtener_matches(des_src:np.ndarray, des_dst:np.ndarray, crossCheck:bool) -> list[cv2.BFMatcher]:
    """
    Obtiene los puntos que matchean entre las imágenes a analizar y comparar

        Parámetros:
            des_src(np.ndarray): array de los descriptores de los keypoints de la imágen de "salida" (en el caso de esta práctica, la imagen de un extremo)
            des_dst(np.ndarray): array de los descriptores de los keypoints de la imágen de "llegada" (en el caso de esta práctica, la imagen del centro, es decir, del ancla)
            crossCheck(bool): booleano que determina si se utiliza o no crossCheck para hallar los matches

        Retorna:
            matches(list de cv2.BFMatcher): lista con los BFMatches hallados entre las imagenes
    """
    # Cross-checking: un match (i,j) solo es válido si j es el mejor vecino de i e i es el mejor vecino de j
    # La norma 2 calcula la distancia euclidea entre los descriptores, la cual es EXACTA
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = crossCheck)
    matches = bf.match(des_src, des_dst)
    matches = sorted(matches, key = lambda m: m.distance)

    return matches

def obtener_matches_flann_lowe(des_src:np.ndarray, des_dst:np.ndarray, ratio:float = 0.75) -> list[cv2.DMatch]:
    """
    Encuentra matches entre dos conjuntos de descriptores usando FLANN, aplicando el test de Lowe's Ratio para
    quedarse solo con los matches confiables: para cada descriptor de origen, busca sus 2 vecinos más cercanos
    en destino, y lo acepta solo si el mejor es notablemente mejor que el segundo 

        Parámetros:
            des_src(np.ndarray): array de los descriptores de los keypoints de la imágen de "salida" (en el caso de esta práctica, la imagen de un extremo)
            des_dst(np.ndarray): array de los descriptores de los keypoints de la imágen de "llegada" (en el caso de esta práctica, la imagen del centro, es decir, del ancla)
            ratio(float): umbral del test de Lowe's Ratio; un match se acepta si distance(mejor) < ratio * distance(segundo mejor)

        Retorna:
            good_matches(list[cv2.DMatch]): matches que pasaron el test, ordenados de mejor a peor (distancia ascendente)
    """
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

def resumen_matches(matches:list[cv2.BFMatcher], nombre:str):
    """
    Crea un diccionario con los valores de distancia obtenidos con un determinado metodo (CrossCheck o FLAN + Lowe)
        
        Parámetros:
            matches(list de cv2.BFMatcher): lista con los BFMatches hallados entre las imagenes
            nombre(str): nombre del método utilizado para hallar los matches
    """
    distancias = np.array([m.distance for m in matches])

    return {
        'metodo': nombre,
        'cantidad': len(matches),
        'distancia_media': np.mean(distancias),
        'distancia_mediana': np.median(distancias),
        'distancia_minima': np.min(distancias),
        'distancia_maxima': np.max(distancias)
    }

def obtener_matches_combinado(desc_src:np.ndarray, desc_dst:np.ndarray, ratio:int = 0.75) -> list[cv2.DMatch]:
    """
    Implementa un método combinando CrossCheck y FLANN + Lowe Ratio con el objetivo de analizar los resultados obtenidos a partir de su combinación
        
        Parámetros:
            des_src(np.ndarray): array de los descriptores de los keypoints de la imágen de "salida" (en el caso de esta práctica, la imagen de un extremo)
            des_dst(np.ndarray): array de los descriptores de los keypoints de la imágen de "llegada" (en el caso de esta práctica, la imagen del centro, es decir, del ancla)
            ratio(float): umbral del test de Lowe's Ratio; un match se acepta si distance(mejor) < ratio * distance(segundo mejor)

        Retorna:
            matches(list[cv2.DMatch]): matches que pasaron el test de Lowe y CrossCheck
                -> matches que pasan el radio de Lowe y se presentan como el mejor vecino j de i y viceversa 
        """
    # Usamos fuerza bruta para aplicar Lowe Ratio en ambos sentidos.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)

    # Calculamos los dos vecinos mas cercanos en ambos sentidos.
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

def mostrar_cambios(img2:np.ndarray, keypoints2:list[cv2.KeyPoint], img1:np.ndarray, keypoints1:list[cv2.KeyPoint], matches_2_1:list[cv2.DMatch], inliers_mask:np.ndarray = None, titulo:str = None, figsize:tuple[int,int] = (14, 6)) -> np.ndarray:
    """
    Dibuja los matches entre dos imágenes. Si no se pasa inliers_mask, dibuja todos los matches con el color
    por defecto de OpenCV. Si se pasa, distingue entre inliers (verde) y outliers (rojo) — útil para
    visualizar el resultado de RANSAC

        Parámetros:
            img2(np.ndarray): imagen de origen (query)
            keypoints2(list[cv2.KeyPoint]): keypoints de img2
            img1(np.ndarray): imagen de destino (train)
            keypoints1(list[cv2.KeyPoint]): keypoints de img1
            matches_2_1(list[cv2.DMatch]): matches entre keypoints2 y keypoints1
            inliers_mask(np.ndarray): máscara booleana/0-1, misma longitud que matches_2_1, indicando qué matches son inliers. Si es None, se dibujan todos los matches por igual
            titulo(str): si se pasa, muestra la figura con este título. Si es None, no grafica, solo devuelve la imagen
            figsize(tuple[int, int]): tamaño de la figura al graficar

        Retorna:
            img_2_1(np.ndarray): imagen resultante con los matches dibujados
    """
    if inliers_mask is None:
        img_2_1 = cv2.drawMatches(img2, keypoints2, img1, keypoints1, matches_2_1, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if titulo is not None:
            plt.figure(figsize = figsize)
            plt.imshow(cv2.cvtColor(img_2_1, cv2.COLOR_BGR2RGB))
            plt.title(titulo)
            plt.axis("off")
            plt.show()
        return img_2_1

    inliers_mask = inliers_mask.astype(int)
    img_2_1 = cv2.drawMatches(img2, [], img1, [], [], None)

    cv2.drawMatches(img2, keypoints2, img1, keypoints1, matches_2_1, outImg = img_2_1,
                    matchesMask = inliers_mask.tolist(), matchColor = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.drawMatches(img2, keypoints2, img1, keypoints1, matches_2_1, outImg = img_2_1, matchesMask = (1 - inliers_mask).tolist(),
    matchColor = (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    if titulo is not None:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(img_2_1, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis("off")
        plt.show()

    return img_2_1

def _dibujar_grupos_puntos(ax:plt.Axes, grupos:list[dict]):
    """
    Dibuja sobre un axis de matplotlib uno o más grupos de puntos, cada uno con su propio color/marcador/label

        Parámetros:
            ax(plt.Axes): eje de matplotlib sobre el que dibujar
            grupos(list[dict]): lista de diccionarios, cada uno con:
                "puntos"(np.ndarray): posiciones (x,y) del grupo
                "color"(str, opcional): color de los puntos
                "label"(str, opcional): etiqueta para la leyenda
                "marker"(str, opcional): tipo de marcador (default "o")
    """
    # grupos: lista de dicts {"puntos": Nx2, "color": str, "label": str opcional}
    if not grupos:
        return
    
    for grupo in grupos:
        puntos = np.asarray(grupo["puntos"])
        color = grupo.get("color", "red")
        label = grupo.get("label")
        marker = grupo.get("marker", "o")
        ax.scatter(puntos[:, 0], puntos[:, 1], c=color, marker = marker, s = 60,
                   edgecolors="black", linewidths=0.8, label = label, zorder = 5)

    if any(g.get("label") for g in grupos):
        ax.legend(loc = "upper right", fontsize = 8)

def mostrar_imagen_con_grilla(img, titulo:str = "", paso:int = 50, figsize:tuple[int,int] = (12, 10), puntos:list[dict] = None, guardar:str = None):
    """
    Muestra una imagen o lista de imagenes (en este caso, en subplots) con una grilla superpuesta, marcando los valores en los ejes, de forma de permitir definir las coordenadas
    a simple vista. Opcionalmente, superpone grupos de puntos

        Parámetros:
            img(np.ndarray | list[np.ndarray]): imagen a mostrar, o lista de imágenes para mostrar una al lado de la otra
            titulo(str | list[str]): título de la imagen, o lista de títulos (uno por imagen) si img es una lista
            paso(int): separación en píxeles entre líneas de la grilla
            figsize(tuple[int, int]): tamaño de la figura
            puntos(list[dict]): grupos de puntos a superponer, o lista de grupos (uno por imagen) si imagen es una lista
            guardar(str): si se especifica, ruta de archivo donde guardar la figura (se crean las carpetas necesarias)
    """
    if guardar is not None:
        os.makedirs(os.path.dirname(guardar), exist_ok=True)

    if isinstance(img, (list, tuple)):
        imagenes = img
        titulos = titulo
        if isinstance(titulos, str):
            titulos = [titulos] * len(imagenes)

        puntos_por_imagen = puntos if puntos is not None else [None] * len(imagenes)

        _, axes = plt.subplots(1, len(imagenes), figsize=figsize)
        if len(imagenes) == 1:
            axes = [axes]

        for ax, imagen, titulo_i, grupos in zip(axes, imagenes, titulos, puntos_por_imagen):
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            ax.imshow(img_rgb)
            ax.set_title(titulo_i)
            ax.set_xticks(np.arange(0, w + 1, paso))
            ax.set_yticks(np.arange(0, h + 1, paso))
            ax.grid(color="yellow", linestyle="-", linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

            _dibujar_grupos_puntos(ax, grupos)

        plt.tight_layout()
        if guardar is not None:
            plt.savefig(guardar, dpi=150, bbox_inches="tight")
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

    _dibujar_grupos_puntos(ax, puntos)

    if guardar is not None:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    plt.show()

def dlt(src:np.ndarray, dst:np.ndarray) -> np.ndarray:
    """
    Calcula la homografía H que mapea exactamente 4 correspondencias de puntos (src -> dst),
    resolviendo directamente el sistema lineal de 8 ecuaciones con 8 incógnitas (cada punto "otorga" 2 filas del sistema)

        Parámetros:
            src(np.ndarray): 4 puntos de origen, de forma (4, 2)
            dst(np.ndarray): 4 puntos de destino correspondientes, de forma (4, 2)

        Retorna:
            H(np.ndarray): matriz de homografía (3x3), con H[2,2] = 1
    """
    A = []
    b = []

    # Construimos el sistema de ecuaciones lineales a partir de las correspondencias
    for i in range(4):
        x, y = src[i]
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

def computar_distancias_reproyeccion(H:np.ndarray, pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    """
    Proyecta los puntos pts1 usando la homografía H y calcula la distancia euclídea entre cada punto proyectado y su correspondiente en pts2

        Parámetros:
            H(np.ndarray): matriz de homografía (3x3)
            pts1(np.ndarray): puntos de origen, de forma (N, 2)
            pts2(np.ndarray): puntos de destino con los que se compara la proyección, de forma (N, 2)

        Retorna:
            dist(np.ndarray): distancia de reproyección de cada punto, de forma (N,)
    """
    # Convertir pts1 a coordenadas homogéneas
    pts1_hom = np.hstack([pts1, np.ones((len(pts1), 1))])
    
    # Proyecta pts1 usando H
    pts1_proy = np.dot(H, pts1_hom.T).T
    pts1_proy = pts1_proy[:, :2] / pts1_proy[:, 2].reshape(-1, 1) #normalizar
    
    dist = np.linalg.norm(pts1_proy - pts2, axis = 1)
    return dist 
    
def error_reproyeccion(H:np.ndarray, pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    """
    Calcula el error de reproyección (RMSE) que produce la homografía H al proyectar pts1 y compararlos contra pts2

        Parámetros:
            H(np.ndarray): matriz de homografía (3x3)
            pts1(np.ndarray): puntos de origen
            pts2(np.ndarray): puntos de destino con los que se compara la proyección

        Retorna:
            rmse(float): raíz del error cuadrático medio de las distancias de reproyección
    """
    dist = computar_distancias_reproyeccion(H, pts1, pts2)
    
    rmse = np.sqrt(np.mean(dist**2))
    return rmse

def homografia_svd(pts1:np.ndarray, pts2:np.ndarray):
    """
    Calcula la homografía H que mapea pts1 a pts2 usando DLT (Direct Linear Transform), resolviendo el sistema homogéneo A·h = 0 
    mediante SVD y tomando el vector singular correspondiente al menor valor singular como solución

        Parámetros:
            pts1(np.ndarray): puntos de origen
            pts2(np.ndarray): puntos de destino correspondientes

        Retorna:
            H(np.ndarray): matriz de homografía (3x3), normalizada para que H[2,2] = 1
    """
    A = []
    for (x, y), (x_prima, y_prima) in zip(pts1, pts2):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima, x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima, y_prima])
    A = np.array(A)

    # SVD (agarramos el vector correspondiente al menor valor singular)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2,2]

def puntos_con_matches(keypoints_src:list[cv2.KeyPoint], keypoints_dst:list[cv2.KeyPoint], matches:list[cv2.DMatch]):
    """
    A partir de una lista de matches, extrae las posiciones (x,y) reales de los keypoints de origen y destino involucrados 
    en cada uno, alineadas por índice

        Parámetros:
            keypoints_src(list[cv2.KeyPoint]): keypoints de la imagen de origen
            keypoints_dst(list[cv2.KeyPoint]): keypoints de la imagen de destino
            matches(list[cv2.DMatch]): matches entre ambos conjuntos de keypoints

        Retorna:
            pts_src(np.ndarray): posiciones (x,y) de los keypoints de origen que matchearon
            pts_dst(np.ndarray): posiciones (x,y) de los keypoints de destino que matchearon
    """
    pts_src = np.float32([keypoints_src[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([keypoints_dst[m.trainIdx].pt for m in matches])
    return pts_src, pts_dst

# Para esta función se utilizó como guía el pseudocodigo provisto
def ransac(pts1:np.ndarray, pts2:np.ndarray, T:int = 1000, umbral:float = 5.0, seed:int = 42):
    """
    Estima la homografía que mapea pts1 a pts2 de forma robusta frente a outliers, usando RANSAC:
    en cada iteración toma 4 correspondencias al azar, calcula H con DLT, y cuenta cuántas de las correspondencias 
    son consistentes con esa H (inliers). 
    Al final, recalcula H usando cuadrados mínimos (SVD) sobre todos los inliers del mejor conjunto encontrado

        Parámetros:
            pts1(np.ndarray): puntos de origen
            pts2(np.ndarray): puntos de destino correspondientes
            T(int): cantidad de iteraciones a probar
            umbral(float): distancia de reproyección máxima para considerar una correspondencia inlier
            seed(int): semilla para asegurar reproducibilidad

        Retorna:
            H_final(np.ndarray): homografía final (3x3), recalculada con todos los inliers
            error_final(float): error de reproyección (RMSE) de H_final sobre los inliers
            mejores_inliers(np.ndarray): máscara booleana, indicando qué correspondencias son inliers
    """
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

def homograficas_a_euclideas(H:np.ndarray, puntos:np.ndarray) -> np.ndarray:
    """
    Proyecta puntos con la homografía H a coordenadas euclideas (x,y), dividiendo por la componente homogenea w
    
        Parámetros:
            H(np.ndarray): matriz de homografía (3x3)
            puntos(np.ndarray): puntos a proyectar

        Retorna:
            (np.ndarray): puntos proyectados en coordenadas (x,y)
    """
    # Convierte a array
    puntos = np.asarray(puntos, dtype = np.float64)
    puntos_homograficos = np.hstack([puntos, np.ones((len(puntos), 1))])
    transformados = (H @ puntos_homograficos.T).T

    x, y, w = transformados.T
    return np.stack([x / w, y / w], axis = 1)

def calcular_size_optimo(imagenes:list[np.ndarray], homografias:list[np.ndarray]) -> tuple[tuple[int,int], list[np.ndarray]]:
    """
    Calcula el tamaño de canvas necesario para que, al aplicar cada homografía a su imagen correspondiente, 
    todas entren completas sin recortarse. Las coordenadas que dieron negativas, son corregidas con una traslación ya que
    el (0,0) se encuentra en la esquina superior izquierda, por lo que es incorrecto tener valores negativos

        Parámetros:
            imagenes(list[np.ndarray]): lista de imágenes a transformar
            homografias(list[np.ndarray]): homografía correspondiente a cada imagen

        Retorna:
            size_canvas(tuple[int, int]): tamaño (ancho, alto) del canvas necesario
            homografias_ajustadas(list[np.ndarray]): las mismas homografías, con la traslación aplicada
    """
    todas_esquinas = []
    for img, H in zip(imagenes, homografias):
        alto, ancho = img.shape[:2]
        esquinas = np.array([[0,0], [ancho, alto], [0, alto], [ancho, 0]], dtype = np.float64)
        todas_esquinas.append(homograficas_a_euclideas(H, esquinas))

    todas_esquinas = np.vstack(todas_esquinas)
    min_xy = np.floor(todas_esquinas.min(axis = 0)).astype(int)
    max_xy = np.ceil(todas_esquinas.max(axis = 0)).astype(int)

    ancho_, alto = max_xy - min_xy

    # Trasladamos ya que se pueden haber obtenido coordenadas negativas
    traslacion = np.array([[1, 0, -min_xy[0]],
                           [0, 1, -min_xy[1]],
                           [0, 0, 1]], dtype = np.float64)

    homografias_ajustadas = [traslacion @ H for H in homografias]
    return (int(ancho_), int(alto)), homografias_ajustadas

def warpear(img:np.ndarray, H:np.ndarray, size_canvas:tuple[int, int]) -> tuple[np.ndarray, cv2.distanceTransform]:
    """
    Warpea una imagen con la homografía H y calcula para cada pixel resultante, un peso segun su distancia al borde de la imagen warpeada. Esto le da mayor
    peso al centro de la imágen

        Parámetros:
            img(np.ndarray): imagen original a warpear
            H(np.ndarray): matriz de homografía (3x3)
            size_canvas(tuple[int, int]): tamaño (ancho, alto) del canvas de salida

        Retorna:
            img_warpeada(np.ndarray): imagen resultante de aplicar la homografía
            peso(np.ndarray): para cada píxel, qué tan lejos está del borde de la imagen warpeada,
            elevado a la 4ta potencia (así los bordes pesan mucho menos que el centro al mezclar)
    """
    img_warpeada = cv2.warpPerspective(img, H, size_canvas)

    # Armamos una mascara blanca (255)
    mascara = np.full(img.shape[:2], 255, dtype = np.uint8)
    mascara = cv2.warpPerspective(mascara, H, size_canvas, flags = cv2.INTER_NEAREST)

    # Distacia al pixel negro (borde) mas cercano
    peso = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    peso = peso ** 4 # Elevamos a la 4 para que los bordes tengan menos peso y no se vean tan marcados --> la imagen final menos borrosa

    return img_warpeada, peso

def construir_imagen(imagenes:list[np.ndarray], homografias:list[np.ndarray]):
    """
    Construye la panorámica final: warpea cada imagen con su homografía y las combina mediante
    un promedio ponderado por píxel usando los pesos, para evitar cortes bruscos en las zonas de superposición

        Parámetros:
            imagenes(list[np.ndarray]): lista de imágenes a transformar
            homografias(list[np.ndarray]): homografía correspondiente a cada imagen
        
        Retorna:
            np.ndarray: imagen panorámica final (uint8)
    """
    size_canvas, homografias_ = calcular_size_optimo(imagenes, homografias)
    ancho, alto = size_canvas[:2]

    # Suma en los 3 canales
    acumulado = np.zeros((alto, ancho, 3), dtype = np.float64)
    sumatoria_pesos = np.zeros((alto, ancho), dtype = np.float64)

    for imagen, H in zip(imagenes, homografias_):
        imagen_warpeada, peso = warpear(imagen, H, size_canvas)

        acumulado += imagen_warpeada.astype(np.float64) * peso[..., None]
        sumatoria_pesos += peso

    suma_pesos = np.where(sumatoria_pesos == 0, 1, sumatoria_pesos)
    # Dividimos para obtener colores validos
    resultado = acumulado / suma_pesos[..., None]
    resultado[suma_pesos == 0] = 0

    return resultado.astype(np.uint8)

def superponer_imagenes(base:np.ndarray, transformada:np.ndarray, alpha:float = 0.5):
    """
    Superpone una imagen transformada sobre una imagen base, mezclándolas con un peso fijo (alpha),
    pero solo en las zonas donde la imagen transformada tiene contenido válido — así no oscurece
    con negro las zonas de la base que el warp no llegó a cubrir

        Parámetros:
            base(np.ndarray): imagen de fondo
            transformada(np.ndarray): imagen a superponer
            alpha(float): peso de la imagen transformada en la mezcla (0 = solo base, 1 = solo transformada)

        Retorna:
            resultado(np.ndarray): imagen base con la transformada mezclada encima, solo donde había contenido
    """
    mascara = transformada.sum(axis = 2) > 0
    mezcla = cv2.addWeighted(base, 1 - alpha, transformada, alpha, 0)

    resultado = base.copy()
    resultado[mascara] = mezcla[mascara]
    return resultado

def graficar_mascara(pesos:list[np.ndarray]):
    """
    Grafica, una al lado de la otra, las máscaras de peso usadas para cada una de las 3 imágenes de un conjunto, 
    normalizadas a escala de grises para poder visualizarlas. Con esto, pixeles con menor peso (más cerca del borde, aparecen en negro,
    más en el centro, en blanco)

        Parámetros:
            pesos(list[np.ndarray]): lista con el peso de cada imagen
    """
    # Graficamos la máscara
    plt.figure(figsize = (15, 5))
    titulos = ["Imagen 0", "Imagen 1 (ancla)", "Imagen 2"]

    for idx, peso in enumerate(pesos):
        peso_norm = cv2.normalize(peso, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        plt.subplot(1, 3, idx + 1)
        plt.imshow(peso_norm, cmap='gray')
        plt.title(titulos[idx])
        plt.axis('off')

    plt.suptitle("Máscaras (pesos) aplicadas a cada imagen")
    plt.tight_layout()
    plt.show()

def comparar_recorte(panorama_bajo, panorama_alto, titulo):
    from i308_utils import show_images  # se instala/importa recien en el notebook

    alto_bajo, ancho_bajo = panorama_bajo.shape[:2]
    alto_alto, ancho_alto = panorama_alto.shape[:2]

    cx_b, cy_b = ancho_bajo // 2, alto_bajo // 2
    w_b, h_b = ancho_bajo // 4, alto_bajo // 4
    recorte_bajo = panorama_bajo[cy_b - h_b:cy_b + h_b, cx_b - w_b:cx_b + w_b]

    cx_a, cy_a = ancho_alto // 2, alto_alto // 2
    w_a, h_a = ancho_alto // 4, alto_alto // 4
    recorte_alto = panorama_alto[cy_a - h_a:cy_a + h_a, cx_a - w_a:cx_a + w_a]

    recorte_bajo_reescalado = cv2.resize(recorte_bajo, (recorte_alto.shape[1], recorte_alto.shape[0]), interpolation=cv2.INTER_NEAREST)
    show_images([recorte_bajo_reescalado, recorte_alto], titles=[f'{titulo} - achicada (reescalada)', f'{titulo} - alta resolución'], figsize=(16, 8))
