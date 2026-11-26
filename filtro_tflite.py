"""Entrenamiento Keras y exportación INT8 para TensorFlow Lite Micro.

No importa PyTorch. Entrada RGB/NHWC normalizada; salida logits de suelo.
Ver TFLITE_S3.md para instalación, comandos y contrato del firmware.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from datos_mixtos import (cargar_archive, preparar_archive, dividir_archive_espacial,
                         validar_particiones, elegir_calibracion, fuente_muestra,
                         dividir_archive_campo_reservado, validar_campo_reservado)
from datos_cropandweed import es_cropandweed, cargar_cropandweed, preparar_cropandweed
from datos_vegann import es_vegann, cargar_vegann, preparar_mezcla_etiquetada
try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "Falta TensorFlow en este intérprete. Usa .venv-tflite\\Scripts\\python.exe "
        "o instala requirements-tflite.txt en un entorno separado."
    ) from exc

RUTA_DS = r"C:/Users/DELL G15/dataset-ninja/maize-tassel-detection/ds0/img"
SEMILLA = 42
_USAR_CACHE_CARGA = False
_CACHE_CARGA = {}
RESOLVERS = {
    "CONV_2D": "AddConv2D", "MAX_POOL_2D": "AddMaxPool2D",
    "RESIZE_NEAREST_NEIGHBOR": "AddResizeNearestNeighbor",
    "CONCATENATION": "AddConcatenation", "RELU": "AddRelu",
    "QUANTIZE": "AddQuantize", "RESHAPE": "AddReshape",
}


@tf.keras.utils.register_keras_serializable(package="Maiz")
class AmpliarNearest(tf.keras.layers.Layer):
    """Resize explícito: UpSampling2D de Keras puede exportar TILE en INT8."""
    def __init__(self, alto, ancho, **kwargs):
        super().__init__(**kwargs)
        self.alto, self.ancho = int(alto), int(ancho)

    def call(self, x):
        return tf.raw_ops.ResizeNearestNeighbor(
            images=x, size=[self.alto, self.ancho],
            align_corners=False, half_pixel_centers=False)

    def get_config(self):
        return {**super().get_config(), "alto": self.alto, "ancho": self.ancho}


def preprocesar(img_bgr, tam=128):
    """Mismas pseudomáscaras ExG+HSV+cierre que filtro.py, sin calcular VARI."""
    if img_bgr is None or not img_bgr.size:
        raise ValueError("Imagen vacía o ilegible")
    img = cv2.resize(img_bgr, (tam, tam), interpolation=cv2.INTER_AREA)
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    suma = r + g + b + 1e-6
    exg = 2.0 * (g / suma) - (r / suma) - (b / suma)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    vegetacion = (exg > 0.01) | ((h >= 18) & (h <= 90) & (s > 30) & (v > 30))
    vegetacion = cv2.morphologyEx(vegetacion.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, (1 - vegetacion).astype(np.float32)[..., None]


def cargar(ruta, tam):
    if isinstance(ruta, np.ndarray):
        ruta = ruta.item()
    clave = (json.dumps(ruta, sort_keys=True) if isinstance(ruta, dict) else os.fsdecode(ruta), tam)
    if _USAR_CACHE_CARGA and clave in _CACHE_CARGA:
        return _CACHE_CARGA[clave]
    if isinstance(ruta, dict):
        resultado = (cargar_vegann(ruta, tam) if es_vegann(ruta) else
                     cargar_cropandweed(ruta, tam) if es_cropandweed(ruta) else cargar_archive(ruta, tam))
    else:
        imagen = cv2.imread(os.fsdecode(ruta))
        if imagen is None:
            raise ValueError(f'Imagen original ilegible: {os.fsdecode(ruta)}')
        resultado = preprocesar(imagen, tam)
    if _USAR_CACHE_CARGA:
        _CACHE_CARGA[clave] = resultado
    return resultado


def buscar_imagenes(carpeta, max_imagenes=None):
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta: {carpeta}")
    rutas = sorted(str(p.resolve()) for p in carpeta.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
    if max_imagenes is not None:
        rutas = rutas[:max_imagenes]
    # Detectar archivos ilegibles antes de dividir, sin conservar imágenes en RAM.
    validas = []
    for indice, ruta in enumerate(rutas):
        imagen = cv2.imread(ruta)
        if imagen is not None and imagen.size:
            validas.append(ruta)
        else:
            print(f"Imagen ilegible, se omite: {ruta}")
        if (indice + 1) % 100 == 0:
            print(f'Original: comprobados {indice + 1}/{len(rutas)} archivos', flush=True)
    if len(validas) < 3:
        raise ValueError("Se necesitan al menos 3 imágenes válidas")
    return validas


def dividir(rutas, semilla=SEMILLA):
    indices = np.random.default_rng(semilla).permutation(len(rutas))
    n = max(1, int(len(rutas) * 0.15))
    return {nombre: [rutas[i] for i in grupo] for nombre, grupo in (
        ("val", indices[:n]), ("test", indices[n:2*n]), ("train", indices[2*n:]))}


def dataset(rutas, tam, batch, aumentar=False, cache=False):
    # JSON conserva rutas originales y registros archive en la misma tubería.
    ds = tf.data.Dataset.from_tensor_slices([json.dumps(r, ensure_ascii=False) for r in rutas])
    if aumentar and not cache:
        ds = ds.shuffle(len(rutas), seed=SEMILLA, reshuffle_each_iteration=True)

    def leer(ruta):
        def leer_json(p):
            if isinstance(p, np.ndarray):
                p = p.item()
            return cargar(json.loads(os.fsdecode(p)), tam)
        x, y = tf.numpy_function(leer_json, [ruta], [tf.float32, tf.float32])
        x.set_shape((tam, tam, 3))
        y.set_shape((tam, tam, 1))
        return x, y

    def aumentos(x, y):
        # Transformar RGB y máscara juntos para mantener la correspondencia.
        conjunto = tf.concat([x, y], axis=-1)
        conjunto = tf.image.random_flip_left_right(conjunto)
        conjunto = tf.image.random_flip_up_down(conjunto)
        conjunto = tf.image.rot90(conjunto, tf.random.uniform([], 0, 4, dtype=tf.int32))
        x, y = conjunto[..., :3], conjunto[..., 3:]
        return tf.clip_by_value(x * tf.random.uniform([], 0.85, 1.15), 0, 1), y

    ds = ds.map(leer, num_parallel_calls=2)
    if cache:
        ds = ds.cache()
    if aumentar:
        if cache:
            ds = ds.shuffle(len(rutas), seed=SEMILLA, reshuffle_each_iteration=True)
        ds = ds.map(aumentos, num_parallel_calls=2)
    opciones = tf.data.Options()
    opciones.threading.private_threadpool_size = 2
    return ds.batch(batch).prefetch(1).with_options(opciones)


def crear_modelo(tam=128, canales=4):
    if tam < 8 or tam % 8 or canales < 1:
        raise ValueError("Tamaño múltiplo de 8 y canales positivos requeridos")
    capas = tf.keras.layers

    def conv(x, c, kernel=3):
        x = capas.Conv2D(c, kernel, padding="same", use_bias=False)(x)
        x = capas.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        return capas.ReLU()(x)

    def bloque(x, c):
        return conv(conv(x, c), c)

    entrada = capas.Input(shape=(tam, tam, 3), name="rgb")
    e1 = bloque(entrada, canales)
    e2 = bloque(capas.MaxPooling2D(2)(e1), canales * 2)
    e3 = bloque(capas.MaxPooling2D(2)(e2), canales * 4)
    x = bloque(capas.MaxPooling2D(2)(e3), canales * 8)
    for salto, c in ((e3, canales * 4), (e2, canales * 2), (e1, canales)):
        x = AmpliarNearest(int(salto.shape[1]), int(salto.shape[2]))(conv(x, c, 1))
        x = bloque(capas.Concatenate()([x, salto]), c)
    salida = capas.Conv2D(1, 1, name="logits_suelo")(x)
    return tf.keras.Model(entrada, salida, name="unet_s3_tflite")


def capacidad_modelo(modelo):
    """Conteos estáticos, batch=1. No mide RAM total ni latencia del ESP32."""
    filtros = next(c.filters for c in modelo.layers if isinstance(c, tf.keras.layers.Conv2D))
    macs = 0
    mayor = 0
    for capa in modelo.layers:
        shape = capa.output.shape
        mayor = max(mayor, int(np.prod([d if d is not None else 1 for d in shape])))
        if isinstance(capa, tf.keras.layers.Conv2D):
            h, w, cout = map(int, shape[1:])
            cin = int(capa.input.shape[-1])
            macs += h*w*cout*(cin//capa.groups)*int(np.prod(capa.kernel_size))
    return {'canales_base': filtros, 'canales_encoder_centro': [filtros*k for k in (1,2,4,8)],
            'parametros_entrenables': int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
            'macs_convoluciones_batch1': macs, 'tensor_individual_mayor_int8_bytes': mayor,
            'ram_total_placa_medida': False}


def perdida_segmentacion(peso):
    def bce_dice(y, logits):
        validos = tf.cast(y >= 0, tf.float32)
        y = tf.clip_by_value(y, 0, 1)
        bce_pixeles = tf.nn.weighted_cross_entropy_with_logits(y, logits, peso)
        bce = tf.math.divide_no_nan(tf.reduce_sum(bce_pixeles * validos), tf.reduce_sum(validos))
        p = tf.sigmoid(logits)
        dados = []
        for prob, real in ((p, y), (1 - p, 1 - y)):
            inter = tf.reduce_sum(prob * real * validos, axis=(1, 2, 3))
            total = tf.reduce_sum((prob + real) * validos, axis=(1, 2, 3))
            dados.append((2 * inter + 1e-6) / (total + 1e-6))
        activos = tf.cast(tf.reduce_sum(validos, axis=(1, 2, 3)) > 0, tf.float32)
        dice_loss = tf.math.divide_no_nan(
            tf.reduce_sum((1 - tf.reduce_mean(tf.stack(dados), axis=0)) * activos),
            tf.reduce_sum(activos))
        return 0.5 * bce + 0.5 * dice_loss
    return bce_dice


@tf.keras.utils.register_keras_serializable(package='Maiz')
class IoUConValidos(tf.keras.metrics.BinaryIoU):
    """Excluir y=-1 de la matriz de confusión usada por Keras."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        pesos = tf.cast(y_true >= 0, self.dtype)
        if sample_weight is not None:
            pesos *= tf.cast(sample_weight, self.dtype)
        return super().update_state(tf.clip_by_value(y_true, 0, 1), y_pred, sample_weight=pesos)


def peso_clases(rutas, tam):
    positivos = negativos = 0
    for indice, r in enumerate(rutas):
        y = cargar(r, tam)[1]
        positivos += int((y == 1).sum())
        negativos += int((y == 0).sum())
        if (indice + 1) % 100 == 0:
            print(f'Balance y caché: {indice + 1}/{len(rutas)} imágenes de train', flush=True)
    if not positivos or not negativos:
        print("AVISO: train tiene una sola clase; revisar pseudomáscaras.")
    return float(np.clip(negativos / max(positivos, 1), 0.1, 10.0))


def metricas(conteos):
    tp, tn, fp, fn = map(int, conteos)
    us, uc = tp + fp + fn, tn + fp + fn
    ious = [v / u for v, u in ((tp, us), (tn, uc)) if u]
    return {"iou_suelo": tp / us if us else 1.0,
            "iou_cultivo": tn / uc if uc else 1.0,
            "miou": sum(ious) / len(ious),
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def cuantizar_entrada(rgb, detalle):
    escala, cero = detalle["quantization"]
    if escala <= 0 or detalle["dtype"] != np.int8:
        raise ValueError("Se esperaba entrada INT8 con escala positiva")
    return np.clip(np.rint(rgb / escala + cero), -128, 127).astype(np.int8)


def inspeccionar_tflite(contenido):
    """Rechaza fallback float/Flex; enumera operadores y versiones reales."""
    from tensorflow.lite.python import schema_py_generated as schema
    interprete = tf.lite.Interpreter(
        model_content=contenido, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]
    for detalle in (entrada, salida):
        if detalle["dtype"] != np.int8 or detalle["quantization"][0] <= 0:
            raise ValueError("La entrada y salida deben estar cuantizadas a INT8")
    if any(np.issubdtype(t["dtype"], np.floating) for t in interprete.get_tensor_details()):
        raise ValueError("La conversión dejó tensores flotantes")
    nombres = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    modelo = schema.Model.GetRootAsModel(contenido, 0)
    operadores = []
    for i in range(modelo.OperatorCodesLength()):
        op = modelo.OperatorCodes(i)
        nombre = nombres.get(op.BuiltinCode(), "DESCONOCIDO")
        if nombre not in RESOLVERS:
            raise ValueError(f"Operador no previsto para el resolver Micro: {nombre}")
        operadores.append({"nombre": nombre, "version": int(op.Version())})
    contrato = {}
    for nombre, detalle in (("entrada", entrada), ("salida", salida)):
        escala, cero = detalle["quantization"]
        contrato[nombre] = {"shape": detalle["shape"].tolist(), "dtype": "int8",
                            "scale": float(escala), "zero_point": int(cero)}
    contrato["operadores"] = operadores
    contrato["tensor_mayor_bytes"] = max(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in interprete.get_tensor_details())
    return interprete, contrato


def exportar_int8(modelo, rutas_calibracion, tam):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    def representativo():
        for ruta in rutas_calibracion:
            x, _ = cargar(ruta, tam)
            yield [x[None].astype(np.float32)]

    # Firma estática y pesos congelados: no exportar recursos READ_VARIABLE,
    # que no corresponden a una red de inferencia autónoma en Micro.
    @tf.function(input_signature=[tf.TensorSpec([1, tam, tam, 3], tf.float32, name="rgb")])
    def inferir(rgb):
        return modelo(rgb, training=False)
    congelada = convert_variables_to_constants_v2(inferir.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([congelada])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representativo
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    contenido = converter.convert()
    inspeccionar_tflite(contenido)
    return contenido


def comparar(modelo, interprete, rutas, tam):
    entrada, salida = interprete.get_input_details()[0], interprete.get_output_details()[0]
    conteos = {"keras_fp32": np.zeros(4, np.int64), "tflite_int8": np.zeros(4, np.int64)}
    distintos = pixeles = 0
    grupos = {}
    for ruta in rutas:
        x, y = cargar(ruta, tam)
        pred_float = modelo(x[None], training=False).numpy()[0, ..., 0] > 0
        interprete.set_tensor(entrada["index"], cuantizar_entrada(x[None], entrada))
        interprete.invoke()
        # El cero REAL de los logits corresponde al zero_point, no siempre al entero 0.
        pred_int = interprete.get_tensor(salida["index"])[0, ..., 0] > salida["quantization"][1]
        real = y[..., 0] > 0.5
        validos = y[..., 0] >= 0
        fuente = fuente_muestra(ruta)
        if fuente not in grupos:
            grupos[fuente] = {'keras_fp32': np.zeros(4, np.int64),
                              'tflite_int8': np.zeros(4, np.int64), 'distintos': 0, 'pixeles': 0}
        for nombre, pred in (("keras_fp32", pred_float), ("tflite_int8", pred_int)):
            c = np.array([np.sum(pred & real & validos), np.sum(~pred & ~real & validos),
                          np.sum(pred & ~real & validos), np.sum(~pred & real & validos)])
            conteos[nombre] += c
            grupos[fuente][nombre] += c
        diferencia = int(np.sum((pred_float != pred_int) & validos))
        cantidad = int(validos.sum())
        distintos += diferencia
        pixeles += cantidad
        grupos[fuente]['distintos'] += diferencia
        grupos[fuente]['pixeles'] += cantidad
    if not pixeles:
        raise ValueError('No hay píxeles válidos en prueba')
    resultado = {nombre: metricas(c) for nombre, c in conteos.items()}
    resultado["fraccion_pixeles_diferentes"] = distintos / pixeles
    resultado['pixeles_validos'] = pixeles
    resultado['por_fuente'] = {
        fuente: {**{nombre: metricas(g[nombre]) for nombre in ('keras_fp32', 'tflite_int8')},
                 'pixeles_validos': g['pixeles'],
                 'fraccion_pixeles_diferentes': g['distintos'] / g['pixeles']}
        for fuente, g in grupos.items() if g['pixeles']}
    return resultado


def escribir_cpp(contenido, salida, contrato):
    (salida / "modelo_suelo.h").write_text(
        "#pragma once\nextern const unsigned char g_modelo_suelo[];\n"
        "extern const unsigned int g_modelo_suelo_len;\n", encoding="utf-8")
    filas = ["  " + ", ".join(f"0x{b:02x}" for b in contenido[i:i+12])
             for i in range(0, len(contenido), 12)]
    (salida / "modelo_suelo.cc").write_text(
        '#include "modelo_suelo.h"\nalignas(16) const unsigned char g_modelo_suelo[] = {\n'
        + ",\n".join(filas) + "\n};\n"
        + f"const unsigned int g_modelo_suelo_len = {len(contenido)};\n", encoding="utf-8")
    nombres = sorted({o["nombre"] for o in contrato["operadores"]})
    lineas = [f'  if (resolver.{RESOLVERS[n]}() != kTfLiteOk) return kTfLiteError;' for n in nombres]
    (salida / "resolver_suelo.h").write_text(
        '#pragma once\n#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
        + f"using ResolverSuelo = tflite::MicroMutableOpResolver<{len(nombres)}>;\n"
        + "inline TfLiteStatus RegistrarOperadoresSuelo(ResolverSuelo& resolver) {\n"
        + "\n".join(lineas) + "\n  return kTfLiteOk;\n}\n", encoding="utf-8")


def validar_protocolo_externo(partes, configuracion, carpeta_externa):
    """Impedir que un entrenamiento mixto se presente como RGB → externo."""
    if configuracion.get('archive2') or configuracion.get('archivo_inicial'):
        raise ValueError('La prueba RGB → archive2 requiere pesos entrenados desde cero solo con el conjunto original')
    externa = Path(carpeta_externa).resolve()
    for muestras in partes.values():
        for m in muestras:
            if ((isinstance(m, dict) and not (es_cropandweed(m) or es_vegann(m)))
                    or Path(m['ruta'] if isinstance(m, dict) else m).resolve().is_relative_to(externa)):
                raise ValueError('Archive (2) debe quedar completamente fuera de train, validación y test RGB')


def main():
    global _USAR_CACHE_CARGA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruta-ds", default=RUTA_DS)
    parser.add_argument("--salida", type=Path, help="Por defecto salida_tflite_rgb_externo")
    parser.add_argument('--archive2', type=Path, default=Path.home()/'Downloads'/'archive (2)')
    parser.add_argument('--cropandweed', type=Path, nargs='?', const=Path('data/cropandweed'),
                        help='Añadir RGB y máscaras labelIds de CropAndWeed a los datasets elegidos')
    parser.add_argument('--max-cropandweed-por-particion', type=int,
                        help='Submuestra reproducible por partición para comprobar el flujo')
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument('--mezcla-etiquetada', action='store_true',
                      help='Mezcla limitada de CropAndWeed + VegAnn, desde cero y solo con máscaras provistas')
    modo.add_argument('--solo-cropandweed', action='store_true',
                      help='Entrenar desde cero solo con RGB y máscaras provistas de CropAndWeed; sin pseudomáscaras')
    modo.add_argument('--solo-original', dest='solo_original', action='store_true',
                      help='Entrenar solo con el dataset original (predeterminado)')
    modo.add_argument('--incluir-archive2', dest='solo_original', action='store_false',
                      help='Experimento mixto opcional; no es una prueba externa')
    modo.add_argument('--mixto-prueba-reservada', action='store_true',
                      help='Mezclar RGB + un campo archive; reservar el otro campo entero para test')
    parser.set_defaults(solo_original=True)
    parser.add_argument('--vegann', type=Path, default=Path.home()/'Downloads'/'VegAnn_dataset'/'VegAnn_dataset')
    parser.add_argument('--train-por-dataset', type=int, default=800,
                        help='Máximo train por fuente en --mezcla-etiquetada')
    parser.add_argument('--val-por-dataset', type=int, default=150)
    parser.add_argument('--test-por-dataset', type=int, default=150)
    parser.add_argument('--campo-prueba-archive', choices=('water', 'rust'), default='water',
                        help='Campo que queda fuera de train/val en --mixto-prueba-reservada')
    parser.add_argument('--margen-espacial', type=int, default=224,
                        help='Separación en píxeles entre train/val del campo archive utilizado para aprender')
    parser.add_argument('--probar-archive2', action='store_true',
                        help='Después de entrenar/exportar, evaluar archive2 sin usarlo para ajustar el modelo')
    parser.add_argument('--cache-datos', action='store_true',
                        help='Cachear imágenes preprocesadas en RAM de la PC, antes de los aumentos aleatorios')
    parser.add_argument('--particiones-originales', type=Path,
                        help='Reutilizar un particiones.json RGB ya verificado, sin repetir el escaneo de JPG')
    parser.add_argument('--entrada-archive', choices=('npy', 'jpg'), default='npy',
                        help='NPY: bandas [2,1,0] en [0,1]; JPG: como se distribuye')
    parser.add_argument('--max-archive-por-particion', type=int, help='Límite por partición para pruebas breves')
    parser.add_argument('--preparar-datos', action='store_true', help='Guardar/verificar particiones sin entrenar')
    parser.add_argument('--inicializar-desde', type=Path, help='Ajustar un .keras previo; sin esto se entrena desde cero')
    parser.add_argument('--lr', type=float, help='Por defecto 1e-4 al ajustar o 1e-3 desde cero')
    parser.add_argument("--tam", type=int, choices=(64, 96, 128), default=128)
    arquitectura = parser.add_mutually_exclusive_group()
    arquitectura.add_argument('--modelo-1mb', action='store_true',
                              help='U-Net de canales 24/48/96/192, aproximadamente 1 MB en INT8; requiere reentrenar')
    arquitectura.add_argument('--canales-base', type=int, help='Filtros del primer nivel; compacto=4, versión ~1 MB=24')
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-imagenes", type=int)
    parser.add_argument("--calibracion", type=int, default=100)
    parser.add_argument("--resumen", action="store_true")
    parser.add_argument("--solo-exportar", type=Path, help="Reexportar un .keras de esta versión sin entrenar")
    args = parser.parse_args()
    arquitectura_explicita = args.modelo_1mb or args.canales_base is not None
    args.canales_base = 24 if args.modelo_1mb else (args.canales_base if args.canales_base is not None else 4)
    if args.canales_base < 1:
        parser.error('--canales-base debe ser positivo')
    if args.mixto_prueba_reservada:
        args.solo_original = False
    if args.mezcla_etiquetada:
        if (args.inicializar_desde or args.solo_exportar or args.particiones_originales
                or args.max_imagenes is not None or args.max_cropandweed_por_particion is not None
                or args.max_archive_por_particion is not None):
            parser.error('--mezcla-etiquetada usa pesos nuevos; limita imágenes con --train/val/test-por-dataset')
        if min(args.train_por_dataset, args.val_por_dataset, args.test_por_dataset) < 1:
            parser.error('Los límites por dataset deben ser positivos')
        if args.cropandweed is None:
            args.cropandweed = Path('data/cropandweed')
    if args.solo_cropandweed:
        if args.inicializar_desde or args.solo_exportar or args.particiones_originales or args.max_imagenes is not None:
            parser.error('--solo-cropandweed requiere entrenamiento desde cero; no combina pesos previos ni particiones originales')
        if args.cropandweed is None:
            args.cropandweed = Path('data/cropandweed')
    if min(args.epocas, args.batch_size, args.calibracion) < 1:
        parser.error("Épocas, batch y calibración deben ser positivos")
    if args.max_imagenes is not None and args.max_imagenes < 3:
        parser.error("--max-imagenes debe ser al menos 3")
    if args.max_archive_por_particion is not None and args.max_archive_por_particion < 1:
        parser.error('--max-archive-por-particion debe ser positivo')
    if args.max_cropandweed_por_particion is not None and (args.max_cropandweed_por_particion < 1 or not args.cropandweed):
        parser.error('--max-cropandweed-por-particion requiere --cropandweed y debe ser positivo')
    if args.cropandweed and args.solo_exportar:
        parser.error('Al reexportar no añadas --cropandweed: se reutiliza el manifiesto guardado')
    if args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0):
        parser.error('--lr debe ser positivo y finito')
    if args.solo_exportar and args.inicializar_desde:
        parser.error('--solo-exportar y --inicializar-desde no se combinan')
    if args.margen_espacial < 0:
        parser.error('--margen-espacial debe ser no negativo')
    if args.mixto_prueba_reservada and args.inicializar_desde:
        parser.error('La prueba con campo reservado se entrena desde cero para no heredar pesos que hayan visto ese campo')
    if args.probar_archive2 and (not args.solo_original or args.inicializar_desde):
        parser.error('--probar-archive2 requiere entrenamiento desde cero solo con RGB; no usar pesos mixtos')
    if args.probar_archive2 and not args.archive2.is_dir():
        parser.error(f'No existe el conjunto externo: {args.archive2}')
    if args.particiones_originales and (args.max_imagenes is not None or args.solo_exportar):
        parser.error('--particiones-originales no se combina con --max-imagenes ni --solo-exportar')
    _USAR_CACHE_CARGA = args.cache_datos
    _CACHE_CARGA.clear()
    if args.salida is None:
        args.salida = (args.solo_exportar.parent if args.solo_exportar else
                       Path('salida_tflite_mixto_reservado' if args.mixto_prueba_reservada else
                            'salida_tflite_rgb_externo' if args.solo_original else 'salida_tflite_mixto'))
        if not args.solo_exportar and args.cropandweed:
            args.salida = (Path('salida_tflite_cw_vegann_limitado') if args.mezcla_etiquetada else
                           Path('salida_tflite_solo_cropandweed') if args.solo_cropandweed else
                           Path('salida_tflite_cropandweed') if args.solo_original else
                           args.salida.with_name(args.salida.name + '_cropandweed'))
        if not args.solo_exportar and args.canales_base != 4:
            args.salida = args.salida.with_name(args.salida.name + f'_c{args.canales_base}')
    tf.keras.utils.set_random_seed(SEMILLA)
    modelo = crear_modelo(args.tam, args.canales_base)
    if args.resumen:
        modelo.summary()
        print(json.dumps(capacidad_modelo(modelo), indent=2))
        print("Entrada RGB/NHWC; salida logits. Resumen FP32, no medición de arena Micro.")
        return
    args.salida.mkdir(parents=True, exist_ok=True)
    manifiesto = args.salida / "particiones.json"
    configuracion = {}
    try:
        if args.solo_exportar:
            # Reutilizar exactamente train/test de la ejecución original: sin fuga de datos.
            partes = json.loads((args.solo_exportar.parent / "particiones.json").read_text(encoding="utf-8"))
            modelo = tf.keras.models.load_model(args.solo_exportar, compile=False)
            if modelo.input_shape[1:] != (args.tam, args.tam, 3):
                raise ValueError("--tam no coincide con el modelo .keras")
            archivo_config = args.solo_exportar.parent/'configuracion_datos.json'
            if archivo_config.is_file():
                configuracion = json.loads(archivo_config.read_text(encoding='utf-8'))
        elif args.mezcla_etiquetada:
            partes, configuracion = preparar_mezcla_etiquetada(
                args.cropandweed, args.vegann, args.train_por_dataset,
                args.val_por_dataset, args.test_por_dataset, SEMILLA)
        else:
            if args.solo_cropandweed:
                partes = {k: [] for k in ('train', 'val', 'test')}
            elif args.particiones_originales:
                partes = json.loads(args.particiones_originales.read_text(encoding='utf-8'))
                validar_particiones(partes)
                raiz_original = Path(args.ruta_ds).resolve()
                for muestras in partes.values():
                    for muestra in muestras:
                        if (not isinstance(muestra, str)
                                or not Path(muestra).resolve().is_relative_to(raiz_original)
                                or not Path(muestra).is_file()):
                            raise ValueError('El manifiesto debe contener solo archivos existentes del dataset RGB original')
                print('Reutilizando particiones RGB originales verificadas.', flush=True)
            else:
                print('Comprobando imágenes del conjunto original...', flush=True)
                partes = dividir(buscar_imagenes(args.ruta_ds, args.max_imagenes))
            configuracion = {'original': {'ruta': args.ruta_ds,
                              'etiquetas': 'Pseudomáscaras ExG+HSV+cierre',
                              'particion': 'Aleatoria determinista, misma semilla y algoritmo anteriores'},
                              'archivo_inicial': str(args.inicializar_desde) if args.inicializar_desde else None}
            if args.solo_cropandweed:
                configuracion.pop('original')
                configuracion['solo_etiquetas_provistas'] = True
            if args.particiones_originales:
                configuracion['original']['particiones_reutilizadas'] = str(args.particiones_originales.resolve())
            if args.cropandweed:
                print('Preparando CropAndWeed con sus máscaras y separación por sesión...', flush=True)
                cw, info_cw = preparar_cropandweed(args.cropandweed, SEMILLA, args.max_cropandweed_por_particion)
                for k in partes:
                    partes[k] += cw[k]
                configuracion['cropandweed'] = info_cw
            if not args.solo_original:
                print('Comprobando bandas y máscaras de archive (2)...', flush=True)
                muestras, info = preparar_archive(args.archive2, args.tam, args.entrada_archive)
                if args.mixto_prueba_reservada:
                    archive, espacial = dividir_archive_campo_reservado(
                        muestras, args.campo_prueba_archive, args.margen_espacial)
                else:
                    archive, espacial = dividir_archive_espacial(muestras)
                if args.max_archive_por_particion is not None:
                    archive = {k: elegir_calibracion(v, args.max_archive_por_particion)
                               for k, v in archive.items()}
                for k in partes:
                    partes[k] += archive[k]
                configuracion['archive2'] = {**info, 'particion_espacial': espacial,
                                            'limite_prueba_por_particion': args.max_archive_por_particion}
            if args.inicializar_desde:
                if args.inicializar_desde.resolve() == (args.salida/'mejor_modelo.keras').resolve():
                    raise ValueError('Usa otra carpeta --salida para conservar el modelo de inicialización')
                modelo = tf.keras.models.load_model(args.inicializar_desde, compile=False)
                if modelo.input_shape[1:] != (args.tam, args.tam, 3) or modelo.output_shape[1:] != (args.tam,args.tam,1):
                    raise ValueError('El modelo inicial no tiene el tamaño y entrada RGB de segmentación esperados')
        validar_particiones(partes)
        capacidad = capacidad_modelo(modelo)
        if arquitectura_explicita and capacidad['canales_base'] != args.canales_base:
            raise ValueError('El .keras cargado no tiene los filtros solicitados; para aumentar filtros hay que entrenar desde cero')
        reserva = configuracion.get('archive2', {}).get('particion_espacial', {}).get('campo_reservado')
        if reserva:
            validar_campo_reservado(partes, reserva)
        if args.probar_archive2:
            validar_protocolo_externo(partes, configuracion, args.archive2)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    manifiesto.write_text(json.dumps(partes, indent=2, ensure_ascii=False), encoding="utf-8")
    recuentos = {}
    for nombre, muestras in partes.items():
        recuentos[nombre] = {}
        for m in muestras:
            fuente = fuente_muestra(m)
            recuentos[nombre][fuente] = recuentos[nombre].get(fuente, 0) + 1
    configuracion['recuentos'] = recuentos
    configuracion['ignorar_etiqueta'] = -1
    configuracion['protocolo'] = ('CropAndWeed + VegAnn limitados; solo máscaras provistas, sin pseudomáscaras'
                                  if configuracion.get('mezcla_etiquetada') else
                                  'Solo CropAndWeed con máscaras provistas; sin pseudomáscaras'
                                  if configuracion.get('solo_etiquetas_provistas') else
                                  'RGB original; archive2 solo como prueba externa tras congelar el modelo'
                                  if args.probar_archive2 else
                                  f'Mixto con campo completo reservado: {reserva}' if reserva else
                                  'Mixto explícito' if configuracion.get('archive2') else 'Solo original')
    if configuracion.get('cropandweed'):
        configuracion['protocolo'] += '; incluye CropAndWeed con sesiones disjuntas y máscaras provistas'
    configuracion['cache_datos_pc'] = args.cache_datos
    configuracion['arquitectura'] = capacidad
    (args.salida/'configuracion_datos.json').write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Imágenes:", {k: len(v) for k, v in partes.items()})
    print('Por fuente:', json.dumps(recuentos, ensure_ascii=False))
    if args.preparar_datos:
        print('Particiones verificadas y guardadas; no se entrenó ni exportó un modelo.')
        return
    checkpoint = args.salida / "mejor_modelo.keras"
    if not args.solo_exportar:
        print('Calculando balance de clases con píxeles válidos de train...', flush=True)
        peso = peso_clases(partes["train"], args.tam)
        print(f"Peso BCE suelo: {peso:.3f}; selección por mIoU de ambas clases.")
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr or (1e-4 if args.inicializar_desde else 1e-3)),
            loss=perdida_segmentacion(peso),
            metrics=[IoUConValidos(target_class_ids=[0, 1], threshold=0.0, name="miou"),
                     IoUConValidos(target_class_ids=[0], threshold=0.0, name="iou_cultivo"),
                     IoUConValidos(target_class_ids=[1], threshold=0.0, name="iou_suelo")])
        historial = modelo.fit(
            dataset(partes["train"], args.tam, args.batch_size, True, cache=args.cache_datos),
            validation_data=dataset(partes["val"], args.tam, args.batch_size, cache=args.cache_datos),
            epochs=args.epocas, shuffle=False,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_miou",
                                                          mode="max", save_best_only=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_miou", mode="max",
                                                           factor=0.5, patience=4, min_lr=1e-5),
                       tf.keras.callbacks.TerminateOnNaN()])
        (args.salida / "historial.json").write_text(
            json.dumps(historial.history, indent=2), encoding="utf-8")
        if not checkpoint.is_file() or not all(np.isfinite(historial.history["loss"])):
            raise RuntimeError("El entrenamiento no produjo un modelo válido")
        modelo = tf.keras.models.load_model(checkpoint, compile=False)

    calibracion = elegir_calibracion(partes['train'], args.calibracion)
    if reserva and any(isinstance(m, dict) and m.get('campo') == reserva for m in calibracion):
        raise ValueError('La calibración no puede utilizar el campo reservado')
    print(f"Calibrando INT8 con {len(calibracion)} imágenes de train, sin aumentos...")
    contenido = exportar_int8(modelo, calibracion, args.tam)
    interprete, contrato = inspeccionar_tflite(contenido)
    resultados = comparar(modelo, interprete, partes["test"], args.tam)
    (args.salida / "modelo_suelo_int8.tflite").write_bytes(contenido)
    escribir_cpp(contenido, args.salida, contrato)
    reporte = {"tensorflow": tf.__version__, "keras": tf.keras.__version__,
               'arquitectura': capacidad_modelo(modelo),
               "parametros_entrenables": int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
               "tflite_bytes": len(contenido), "tam": args.tam, "semilla": SEMILLA,
               "contrato": contrato, "resultados_test": resultados,
               "calibracion": calibracion,
               "normalizacion": ("RGB/NHWC; original y JPG: uint8 / 255; NPY: [2,1,0] ya en [0,1]"
                                  if configuracion.get('archive2') else "RGB/NHWC, uint8 / 255.0"),
               "suelo": "salida_int8 > zero_point_salida", "cultivo": "salida_int8 <= zero_point_salida",
               "etiquetas": ("Original: ExG+HSV; archive2: máscaras provistas, ID 0=suelo/fondo; -1 se ignora"
                              if configuracion.get('archive2') else "Solo original: pseudomáscaras ExG+HSV+cierre"),
               'configuracion_datos': configuracion,
               'resolucion_metricas': [args.tam,args.tam],
               "hardware_validado": False, "tensor_arena_bytes": None}
    if configuracion.get('cropandweed'):
        etiquetas_cw = 'CropAndWeed: ID 0=suelo, 1..99 y 255=vegetación; otros ignorados'
        reporte['etiquetas'] = (etiquetas_cw if configuracion.get('solo_etiquetas_provistas')
                                else reporte['etiquetas'] + '; ' + etiquetas_cw)
    if configuracion.get('vegann'):
        reporte['etiquetas'] += '; VegAnn: 0=fondo (suelo/residuos), 255=vegetación'
    if reserva:
        reporte['prueba_campo_reservado'] = {
            'campo': reserva, 'usado_para_entrenar': False,
            'usado_para_validar': False, 'usado_para_calibrar': False,
            'resultados': resultados['por_fuente'][f'archive2/{reserva}']}
    (args.salida / "reporte.json").write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultados, indent=2))
    print(f"TFLite INT8: {len(contenido):,} bytes; archivos en {args.salida.resolve()}")
    print("Pendiente: comprobar versiones de kernels, AllocateTensors y latencia en ESP32-S3.")
    if args.probar_archive2:
        print('Modelo y umbral congelados. Iniciando evaluación externa de archive (2)...', flush=True)
        carpeta_externa = args.salida/'evaluacion_externa_archive2'
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluar_archive2_tflite.py')),
                        '--dataset', str(args.archive2.resolve()),
                        '--modelo', str((args.salida/'modelo_suelo_int8.tflite').resolve()),
                        '--salida', str(carpeta_externa.resolve())], check=True)
        externo = json.loads((carpeta_externa/'reporte.json').read_text(encoding='utf-8'))
        reporte['prueba_externa_archive2'] = {
            'usado_para_entrenar': False, 'usado_para_validar': False,
            'usado_para_calibrar': False, 'reporte': str((carpeta_externa/'reporte.json').resolve()),
            'evaluacion': externo['evaluacion']}
        (args.salida/'reporte.json').write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == "__main__":
    main()
