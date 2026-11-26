"""CropAndWeed original: RGB + labelIds, con particiones por sesión."""
from pathlib import Path
import re

import cv2
import numpy as np


def es_cropandweed(m):
    return isinstance(m, dict) and m.get('fuente') == 'cropandweed'


def leer_cropandweed_nativo(muestra):
    imagen = cv2.imread(muestra['ruta'])
    mascara = cv2.imread(muestra['mascara'], cv2.IMREAD_UNCHANGED)
    if imagen is None or mascara is None or mascara.ndim != 2 or imagen.shape[:2] != mascara.shape:
        raise ValueError(f"Pareja CropAndWeed ilegible/no alineada: {muestra['ruta']}")
    # Es el mapa base CropAndWeed, no una variante remapeada como Maize1/Fine24.
    # 255 es Vegetation genérica, NO es una etiqueta para ignorar.
    validos = (mascara <= 99) | (mascara == 255)
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, (mascara == 0).astype(np.float32), validos


def cargar_cropandweed(muestra, tam):
    rgb, suelo, validos = leer_cropandweed_nativo(muestra)
    x = cv2.resize(rgb, (tam, tam), interpolation=cv2.INTER_AREA)
    y = cv2.resize(suelo, (tam, tam), interpolation=cv2.INTER_NEAREST)
    soporte = cv2.resize(validos.astype(np.float32), (tam, tam), interpolation=cv2.INTER_AREA)
    return x, np.where(soporte > 1-1e-6, y, -1).astype(np.float32)[..., None]


def dividir_por_sesion(muestras, semilla=42):
    """Separar sesiones completas, estratificando por origen ave/vwg."""
    partes = {k: [] for k in ('train', 'val', 'test')}
    sesiones = {}
    for m in muestras:
        sesiones.setdefault(m['sesion'], []).append(m)
    rng = np.random.default_rng(semilla)
    for prefijo in sorted({s.split('-')[0] for s in sesiones}):
        grupo = sorted(s for s in sesiones if s.split('-')[0] == prefijo)
        if len(grupo) < 3:
            raise ValueError(f'Se necesitan al menos 3 sesiones del origen {prefijo}')
        orden = rng.permutation(len(grupo))
        n = max(1, int(.15*len(grupo)))
        for nombre, indices in (('val', orden[:n]), ('test', orden[n:2*n]), ('train', orden[2*n:])):
            for i in indices:
                partes[nombre].extend(sesiones[grupo[i]])
    for nombre in partes:
        partes[nombre].sort(key=lambda m: m['ruta'])
    validar_sesiones(partes)
    return partes


def validar_sesiones(partes):
    vistas = {}
    for nombre, muestras in partes.items():
        for m in muestras:
            if not es_cropandweed(m):
                continue
            sesion = m['sesion']
            anterior = vistas.setdefault(sesion, nombre)
            if anterior != nombre:
                raise ValueError(f'La sesión CropAndWeed {sesion} aparece en {anterior} y {nombre}')


def preparar_cropandweed(carpeta, semilla=42, limite_por_particion=None):
    base = Path(carpeta).resolve()
    carpeta_masks = base/'labelIds'/'CropAndWeed'
    if not (base/'images').is_dir() or not carpeta_masks.is_dir():
        raise ValueError('CropAndWeed debe contener images y labelIds/CropAndWeed (mapa original)')
    imagenes = {p.stem: p for p in (base/'images').glob('*.jpg')}
    mascaras = {p.stem: p for p in carpeta_masks.glob('*.png')}
    if not imagenes or imagenes.keys() != mascaras.keys():
        raise ValueError(f'Parejas incompletas: {len(imagenes.keys()-mascaras.keys())} sin máscara, '
                         f'{len(mascaras.keys()-imagenes.keys())} sin imagen')
    muestras = []
    for nombre, ruta in sorted(imagenes.items()):
        m = re.fullmatch(r'(ave|vwg)-(\d{4})-(\d{4})', nombre)
        if not m:
            raise ValueError(f'No se puede identificar la sesión: {nombre}')
        muestras.append({'ruta': str(ruta.resolve()), 'mascara': str(mascaras[nombre].resolve()),
                         'fuente': 'cropandweed', 'sesion': '-'.join(m.group(1, 2)),
                         'mapa_clases': 'CropAndWeed_original'})
    partes = dividir_por_sesion(muestras, semilla)
    if limite_por_particion is not None:
        if limite_por_particion < 1:
            raise ValueError('El límite CropAndWeed debe ser positivo')
        rng = np.random.default_rng(semilla)
        for nombre, grupo in partes.items():
            if len(grupo) > limite_por_particion:
                indices = sorted(rng.choice(len(grupo), limite_por_particion, replace=False))
                partes[nombre] = [grupo[i] for i in indices]
    validar_sesiones(partes)
    info = {'base': str(base), 'parejas_disponibles': len(muestras), 'semilla': semilla,
            'separacion': 'Sesiones completas estratificadas por ave/vwg; aproximadamente 70/15/15 de sesiones',
            'etiquetas': {'0': 'suelo -> y=1', '1..99': 'vegetación -> y=0',
                          '255': 'vegetación genérica -> y=0', '100..254': 'no documentados -> ignorar y=-1'},
            'sesiones': {k: sorted({m['sesion'] for m in v}) for k, v in partes.items()},
            'recuentos': {k: len(v) for k, v in partes.items()},
            'limite_prueba_por_particion': limite_por_particion,
            'nota': 'Incluye varias especies, no solo maíz; imágenes/máscaras se decodifican y validan al cargarse'}
    return partes, info
