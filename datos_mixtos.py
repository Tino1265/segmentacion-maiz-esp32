"""Carga y separación espacial de archive (2), sin depender de TensorFlow."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from datos_cropandweed import es_cropandweed, leer_cropandweed_nativo, validar_sesiones
from datos_vegann import es_vegann, leer_vegann_nativo

CLASES_ARCHIVE = {'soil': 0, 'low_stress': 1, 'high_stress': 2, 'healthy': 3, 'rust': 4}


def ruta_muestra(muestra):
    return muestra['ruta'] if isinstance(muestra, dict) else str(muestra)


def fuente_muestra(muestra):
    if es_vegann(muestra):
        return 'vegann'
    if es_cropandweed(muestra):
        return 'cropandweed'
    return f"archive2/{muestra['campo']}" if isinstance(muestra, dict) else 'original'


def leer_muestra_nativa(muestra):
    if es_vegann(muestra):
        return leer_vegann_nativo(muestra)
    return leer_cropandweed_nativo(muestra) if es_cropandweed(muestra) else leer_archive_nativo(muestra)


def leer_archive_nativo(muestra):
    """Devuelve RGB [0,1], suelo binario y validez a resolución nativa."""
    a = np.load(muestra['npy_referencia'], allow_pickle=False).astype(np.float32)
    if a.ndim != 3 or a.shape[-1] != 6:
        raise ValueError(f"Se esperaban 6 bandas: {muestra['npy_referencia']}")
    visible = a[..., :3]
    validos = (np.isfinite(visible).all(-1) & (visible > -9999).all(-1)
               & (visible != 0).any(-1))
    if np.any(visible[validos] < 0) or np.any(visible[validos] > 1):
        raise ValueError('Las bandas visibles deben estar en [0,1]')
    ruta = Path(muestra['ruta'])
    if ruta.suffix.lower() == '.npy':
        bandas = muestra.get('bandas_rgb', [2, 1, 0])
        if sorted(bandas) != [0, 1, 2]:
            raise ValueError('La entrada de este modelo debe usar las tres bandas visibles')
        rgb = np.where(validos[..., None], a[..., bandas], 0)
    else:
        bgr = cv2.imread(str(ruta))
        if bgr is None or bgr.shape[:2] != a.shape[:2]:
            raise ValueError(f'JPG ilegible o no alineado: {ruta}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.where(validos[..., None], rgb, 0)
    mascara = cv2.imread(muestra['mascara'], cv2.IMREAD_UNCHANGED)
    if (mascara is None or mascara.shape != a.shape[:2]
            or not np.isin(mascara, list(CLASES_ARCHIVE.values())).all()):
        raise ValueError(f"Máscara ilegible o con clases no previstas: {muestra['mascara']}")
    # El modelo usa 1=suelo; este dataset usa ID 0=suelo/fondo.
    return rgb.astype(np.float32), (mascara == 0).astype(np.float32), validos


def cargar_archive(muestra, tam):
    rgb, suelo, validos = leer_archive_nativo(muestra)
    rgb = cv2.resize(rgb, (tam, tam), interpolation=cv2.INTER_AREA)
    suelo = cv2.resize(suelo, (tam, tam), interpolation=cv2.INTER_NEAREST)
    soporte = cv2.resize(validos.astype(np.float32), (tam, tam), interpolation=cv2.INTER_AREA)
    # -1 es ignorado por la pérdida, las métricas y el balance de clases.
    y = np.where(soporte > 1 - 1e-6, suelo, -1).astype(np.float32)[..., None]
    return rgb, y


def preparar_archive(carpeta, tam=128, entrada='npy'):
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise ValueError(f'No existe archive (2): {carpeta}')
    mapas = list(carpeta.rglob('meta/class_map.json'))
    if len(mapas) != 1:
        raise ValueError('Se necesita una única carpeta de recortes con meta/class_map.json')
    base = mapas[0].parent.parent
    if json.loads(mapas[0].read_text(encoding='utf-8'))['class_map'] != CLASES_ARCHIVE:
        raise ValueError('El mapa de clases de archive (2) es diferente del esperado')
    if entrada not in ('npy', 'jpg'):
        raise ValueError('Entrada archive debe ser npy o jpg')
    muestras, vacias = [], []
    with (base / 'meta/patches.csv').open(encoding='utf-8-sig', newline='') as f:
        filas = list(csv.DictReader(f))
    for indice, fila in enumerate(filas):
        nombre = fila['id']
        muestra = {
            'ruta': str((base / 'images' / f'{nombre}.{entrada}').resolve()),
            'npy_referencia': str((base / 'images' / f'{nombre}.npy').resolve()),
            'mascara': str((base / 'masks' / f'{nombre}.png').resolve()),
            'fuente': 'archive2', 'campo': fila['src_task'],
            'bandas_rgb': [2, 1, 0] if entrada == 'npy' else None,
            **{k: int(fila[k]) for k in ('x0', 'y0', 'x1', 'y1')},
        }
        if muestra['x1'] <= muestra['x0'] or muestra['y1'] <= muestra['y0']:
            raise ValueError(f'Coordenadas inválidas: {nombre}')
        _, y = cargar_archive(muestra, tam)
        if not np.any(y >= 0):
            vacias.append(nombre)
        else:
            muestras.append(muestra)
        if (indice + 1) % 200 == 0:
            print(f'Archive: comprobados {indice + 1}/{len(filas)} recortes y máscaras', flush=True)
    if len({m['ruta'] for m in muestras}) != len(muestras):
        raise ValueError('Hay recortes duplicados en patches.csv')
    return muestras, {'base': str(base.resolve()), 'entrada': entrada,
                      'inspeccionadas': len(filas), 'sin_senal_excluidas': vacias,
                      'bandas_rgb_npy': [2, 1, 0],
                      'orden_bandas': 'Interpretación RGB inferida del dataset; no confirmada por el sensor'}


def dividir_archive_espacial(muestras):
    """Franjas por campo: 70/15/15 de altura; excluir recortes que crucen límites.

    Los rectángulos de distintas particiones no comparten píxeles del ortomosaico.
    Sigue siendo una evaluación de otras zonas de los mismos campos, no campos nuevos.
    """
    partes = {k: [] for k in ('train', 'val', 'test')}
    campos = {}
    descartadas = []
    for campo in sorted({m['campo'] for m in muestras}):
        grupo = sorted((m for m in muestras if m['campo'] == campo), key=lambda m: m['ruta'])
        alto = max(m['y1'] for m in grupo)
        corte1, corte2 = int(0.70 * alto), int(0.85 * alto)
        conteos = {k: 0 for k in partes}
        for m in grupo:
            if m['y1'] <= corte1:
                nombre = 'train'
            elif m['y0'] >= corte1 and m['y1'] <= corte2:
                nombre = 'val'
            elif m['y0'] >= corte2:
                nombre = 'test'
            else:
                descartadas.append(m['ruta'])
                continue
            partes[nombre].append(m)
            conteos[nombre] += 1
        if not all(conteos.values()):
            raise ValueError(f'El campo {campo} no cubre las tres franjas con señal útil')
        campos[campo] = {'alto': alto, 'cortes_y': [corte1, corte2], 'recuentos': conteos}
    if not campos:
        raise ValueError('No hay recortes archive con señal útil')
    return partes, {'metodo': 'Franjas espaciales 70/15/15 por campo; se excluyen recortes que cruzan cortes',
                    'campos': campos, 'cruzan_limites_excluidos': descartadas}


def dividir_archive_campo_reservado(muestras, campo_prueba='water', margen=224):
    """Un campo entero solo en test; otros campos en franjas train/val 80/20.

    El margen separa espacialmente train y val dentro de cada campo restante.
    Ningún recorte del campo de prueba entra al entrenamiento ni a la validación.
    """
    campos_disponibles = {m['campo'] for m in muestras}
    if campo_prueba not in campos_disponibles or len(campos_disponibles) < 2:
        raise ValueError('Se necesita el campo reservado y al menos otro campo para entrenar')
    if margen < 0:
        raise ValueError('El margen espacial no puede ser negativo')
    partes = {k: [] for k in ('train', 'val', 'test')}
    partes['test'] = sorted((m for m in muestras if m['campo'] == campo_prueba), key=lambda m: m['ruta'])
    detalles, descartadas = {}, []
    for campo in sorted(campos_disponibles - {campo_prueba}):
        grupo = sorted((m for m in muestras if m['campo'] == campo), key=lambda m: m['ruta'])
        alto = max(m['y1'] for m in grupo)
        corte = int(.8 * alto)
        fin_train = corte - margen // 2
        inicio_val = corte + (margen - margen // 2)
        cuentas = {'train': 0, 'val': 0}
        for m in grupo:
            if m['y1'] <= fin_train:
                nombre = 'train'
            elif m['y0'] >= inicio_val:
                nombre = 'val'
            else:
                descartadas.append(m['ruta'])
                continue
            partes[nombre].append(m)
            cuentas[nombre] += 1
        if not all(cuentas.values()):
            raise ValueError(f'El campo {campo} no tiene señal suficiente a ambos lados del margen')
        detalles[campo] = {'alto': alto, 'corte_y': corte, 'fin_train_y': fin_train,
                           'inicio_val_y': inicio_val, 'recuentos': cuentas}
    validar_particiones(partes)
    return partes, {'metodo': 'Campo completo reservado para test; otros campos train/val 80/20 con margen',
                    'campo_reservado': campo_prueba, 'margen_pixels': margen,
                    'campos_train_val': detalles, 'cruzan_limites_excluidos': descartadas}


def validar_campo_reservado(partes, campo):
    """Comprobar la reserva también al cargar un manifiesto para reexportar."""
    for nombre in ('train', 'val'):
        if any(isinstance(m, dict) and m.get('campo') == campo for m in partes[nombre]):
            raise ValueError(f'El campo reservado {campo} aparece en {nombre}')
    if not any(isinstance(m, dict) and m.get('campo') == campo for m in partes['test']):
        raise ValueError(f'El campo reservado {campo} no aparece en test')


def validar_particiones(partes):
    if set(partes) != {'train', 'val', 'test'} or not all(partes.values()):
        raise ValueError('Particiones incompletas')
    ids = [{str(Path(ruta_muestra(m)).resolve()).casefold() for m in partes[k]}
           for k in ('train', 'val', 'test')]
    for k, conjunto in zip(('train', 'val', 'test'), ids):
        if len(conjunto) != len(partes[k]):
            raise ValueError(f'Hay imágenes duplicadas dentro de {k}')
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('Hay imágenes compartidas entre particiones')
    validar_sesiones(partes)
    # Comprobar también rectángulos, no solo nombres de archivo distintos.
    for i, a in enumerate(('train', 'val', 'test')):
        for b in ('train', 'val', 'test')[i+1:]:
            for ma in partes[a]:
                if not isinstance(ma, dict) or 'campo' not in ma:
                    continue
                for mb in partes[b]:
                    if not isinstance(mb, dict) or 'campo' not in mb or ma['campo'] != mb['campo']:
                        continue
                    if (max(ma['x0'], mb['x0']) < min(ma['x1'], mb['x1'])
                            and max(ma['y0'], mb['y0']) < min(ma['y1'], mb['y1'])):
                        raise ValueError('Recortes archive solapados entre particiones')


def elegir_calibracion(muestras, cantidad, semilla=42):
    """Round-robin por fuente/campo: no calibrar solo con el primer dataset."""
    rng = np.random.default_rng(semilla)
    grupos = {}
    for m in muestras:
        grupos.setdefault(fuente_muestra(m), []).append(m)
    for grupo in grupos.values():
        rng.shuffle(grupo)
    seleccion = []
    while len(seleccion) < min(cantidad, len(muestras)):
        for nombre in sorted(grupos):
            if grupos[nombre] and len(seleccion) < cantidad:
                seleccion.append(grupos[nombre].pop())
    return seleccion
