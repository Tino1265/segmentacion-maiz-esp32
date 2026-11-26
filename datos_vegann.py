"""VegAnn: máscaras provistas y mezcla pequeña con CropAndWeed, sin pseudomáscaras."""
import csv
from pathlib import Path

import cv2
import numpy as np

from datos_cropandweed import preparar_cropandweed


def es_vegann(m):
    return isinstance(m, dict) and m.get('fuente') == 'vegann'


def vegetacion_vegann(mascara):
    if mascara.dtype != np.uint8 or mascara.ndim != 2:
        raise ValueError('La máscara VegAnn debe ser gris uint8 en [0,255]')
    # Algunas anotaciones distribuidas tienen bordes grises por interpolación.
    # Binarizar la máscara provista; no calcular un filtro a partir de la foto.
    return mascara >= 128


def leer_vegann_nativo(m):
    bgr = cv2.imread(m['ruta'])
    mascara = cv2.imread(m['mascara'], cv2.IMREAD_UNCHANGED)
    if bgr is None or mascara is None or mascara.ndim != 2 or bgr.shape[:2] != mascara.shape:
        raise ValueError(f"Pareja VegAnn ilegible/no alineada: {m['ruta']}")
    vegetacion = vegetacion_vegann(mascara)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, (~vegetacion).astype(np.float32), np.ones(mascara.shape, bool)


def cargar_vegann(m, tam):
    rgb, suelo, _ = leer_vegann_nativo(m)
    return (cv2.resize(rgb, (tam, tam), interpolation=cv2.INTER_AREA),
            cv2.resize(suelo, (tam, tam), interpolation=cv2.INTER_NEAREST)[..., None])


def preparar_vegann(carpeta):
    base = Path(carpeta).resolve()
    partes = {k: [] for k in ('train', 'val', 'test')}
    nombres = {'Training': 'train', 'Validation': 'val', 'Test': 'test'}
    with (base/'VegAnn_dataset.csv').open(encoding='utf-8-sig', newline='') as f:
        filas = list(csv.DictReader(f, delimiter=';'))
    excluidas = 0
    for fila in filas:
        # Evitar fotos hemisféricas que incluyen cielo como fondo y vistas oblicuas.
        if fila['Orientation'] != 'Nadir':
            excluidas += 1
            continue
        nombre = fila['Name']
        if Path(nombre).name != nombre:
            raise ValueError(f'Nombre VegAnn inválido: {nombre}')
        imagen, mascara = base/'images'/nombre, base/'annotations'/nombre
        if not imagen.is_file() or not mascara.is_file():
            raise ValueError(f'Falta imagen o máscara VegAnn: {nombre}')
        partes[nombres[fila['TVT-split1']]].append({
            'ruta': str(imagen), 'mascara': str(mascara), 'fuente': 'vegann',
            'especie': fila['Species'], 'origen': fila['Dataset-Name'],
            'particion_oficial': fila['TVT-split1']})
    if not all(partes.values()):
        raise ValueError('VegAnn no tiene imágenes nadir en las tres particiones')
    return partes, {'base': str(base), 'particion': 'TVT-split1 oficial, solo Orientation=Nadir',
                    'recuentos_disponibles': {k: len(v) for k, v in partes.items()},
                    'otras_orientaciones_excluidas': excluidas,
                    'etiquetas': '0=fondo (suelo y residuos); 255=vegetación; bordes grises binarizados a 128'}


def seleccionar_cobertura(muestras, limite, semilla=42):
    """Cuotas por cobertura nativa: <20%, 20..60%, >60% de vegetación.

    Seleccionar solo dentro de cada partición, sin reemplazo. Si un estrato es
    pequeño, repartir los lugares restantes entre los otros estratos.
    """
    if limite < 1:
        raise ValueError('El límite debe ser positivo')
    grupos = [[], [], []]
    for m in sorted(muestras, key=lambda m: m['ruta']):
        a = cv2.imread(m['mascara'], cv2.IMREAD_UNCHANGED)
        if a is None or a.ndim != 2:
            raise ValueError(f"Máscara ilegible: {m['mascara']}")
        if es_vegann(m):
            validos, vegetacion = np.ones(a.shape, bool), vegetacion_vegann(a)
        else:
            validos = (a <= 99) | (a == 255)
            vegetacion = validos & (a != 0)
        if not validos.any():
            raise ValueError(f"Máscara sin etiquetas válidas: {m['mascara']}")
        cobertura = float(vegetacion.sum() / validos.sum())
        indice = 0 if cobertura < .2 else 1 if cobertura <= .6 else 2
        grupos[indice].append({**m, 'fraccion_vegetacion': cobertura})
    rng = np.random.default_rng(semilla)
    for g in grupos:
        rng.shuffle(g)
    seleccion = []
    while len(seleccion) < min(limite, len(muestras)):
        for g in grupos:
            if g and len(seleccion) < limite:
                seleccion.append(g.pop())
    seleccion.sort(key=lambda m: m['ruta'])
    return seleccion


def preparar_mezcla_etiquetada(cropandweed, vegann, train=800, val=150, test=150, semilla=42):
    cw, info_cw = preparar_cropandweed(cropandweed, semilla)
    va, info_va = preparar_vegann(vegann)
    partes = {k: [] for k in cw}
    resumen = {}
    for fuente, datos in (('cropandweed', cw), ('vegann', va)):
        resumen[fuente] = {}
        for k, limite in (('train', train), ('val', val), ('test', test)):
            print(f'Seleccionando {fuente}/{k}: hasta {limite} imágenes por cobertura...', flush=True)
            elegidas = seleccionar_cobertura(datos[k], limite, semilla)
            partes[k].extend(elegidas)
            resumen[fuente][k] = {
                'imagenes': len(elegidas),
                'cobertura_baja': sum(m['fraccion_vegetacion'] < .2 for m in elegidas),
                'cobertura_media': sum(.2 <= m['fraccion_vegetacion'] <= .6 for m in elegidas),
                'cobertura_alta': sum(m['fraccion_vegetacion'] > .6 for m in elegidas)}
    return partes, {'cropandweed': info_cw, 'vegann': info_va,
                    'archivo_inicial': None, 'solo_etiquetas_provistas': True,
                    'mezcla_etiquetada': {'limites_por_fuente': {'train': train, 'val': val, 'test': test},
                        'seleccion': 'Semilla fija, sin reemplazo, cuotas por cobertura en cada partición',
                        'resumen': resumen}}
