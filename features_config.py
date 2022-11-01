"""Archivo de configuración del script features.py.

Todas las variables globales que se emplean en el script se definen y modifican 
aquí.

Este script lee una serie de de archivos .wav y .TextGrid de una carpeta de 
entrada y crea un fichero de salida .csv con los siguientes resultados:

    - Datos del locutor del audio.
    - Numeración de las ventanas de audio.
    - Descriptores calculados.
    - Parámetros extraidos del archivo textgrid.
    - Clase de los fonemas.

Si la base de datos ya existe se añaden los nuevos locutores al archivo 
existente.

La guía de estilos que se emplea es la definida por Google para Python.
https://google.github.io/styleguide/pyguide.html#s3.2-line-length

Se ha empleado yapf para corregir el estilo.
https://github.com/google/yapf/
"""

import os


# Entrada
# AUDIO_FOLDER: carpeta donde se almacenan todos los conjuntos de audios.
# AUDIO_DATABASE: carpeta con el conjunto de audios que vamos a procesar.
AUDIO_FOLDER = "audio_database"     
AUDIO_DATABASE = "audios25m_test"


# Salida
# RESULT_FOLDER: carpeta donde se guardan los resultados
# RESULT_DATABASE: nombre del fichero de salida.
RESULT_FOLDER = "resultadosBaseDatos"
RESULT_DATABASE = "audios25m_test_andres.csv"


# Parámetros de ventanas de audio en ms.
# WIN_LENGTH: longitud de la ventana.
# WIN_OVERLAP: superposición (overlap) de la ventana. El paso de cada ventana se
# calcula como (WIN_LENGTH-WIN_OVERLAP)
WIN_LENGTH = 25  
WIN_OVERLAP = 10


# Locutor.
# Se definen los nombres de los parámetros que se extraen del nombre de cada 
# uno de los locutores. Cada parámetro está separado por '_'.
#
# Ej: un audio con el nombre nas20001_iphone_toma1.wav tiene tres partes, que se
# podrían denominar como ["speaker", "device", "numero_toma"].
#
# OJO: Uno de los nombres tiene que ser "speaker" obligatoriamente.
COLUMNS_NAMES_SPEAKER = ["speaker", "device"]  


# Numeración de las ventanas.
# NUM_ZEROS_WINDOWS_MAX: Longitud de la numeración, si es igual a 3 se enumeran
# 001, 002, 003, ..., 999.
# WIN_COLUMN_NAME: nombre de la columna en el archivo de salida final.
NUM_ZEROS_WINDOWS_MAX = 5
WIN_COLUMN_NAME = ["window"]


# Descriptores.
# Los descriptores que se pueden calcular:
# "mfcc"                    mfcc
# TODO"mfcc_librosa"        mfcc librosa    
# "delta"                   delta mfcc
# "deltaDelta"              delta delta
# TODO "vlhr"               6 mfcc con frecuencia de corte 400,500,...,900 Hz
# TODO "formants"           3 formantes frecuencia f1, f2, f3 Hz
# TODO "formantsBW"         3 ancho de banda formantes Hz
# TODO "formantsDist"       3 distancia entre formantes f1-f2, f1-f3, f2-f3 Hz
# TODO "f0"                 3 frecuencia fundamental f0, 2xf0, 3xf0
# TODO "intensity"          1 intensidad en dB como Praat
FEATURES_TO_CALCULATE = ["mfcc", "delta", "deltaDelta"]

# MFCC 
#https://python-speech-features.readthedocs.io/en/latest/index.html
NUMCEP = 13
NFILT=26
NFFT=1024
LOWFREQ=0
HIGHFREQ=None
PREEMPH=0.97
CEPLIFTER=22
APPENDENERGY=True
# TODO: Incluir el resto de parámetros que se pueden configurar aquí

# MFCC librosa
# TODO(mfcc librosa): Incluir la opción de mfcc con librosa
# https://librosa.org/doc/latest/index.html

# Delta y DeltaDelta
DIST_DELTA = 2


# TextGrid 
# TEXTGRID_TIER: Nombre de los tiers disponibles en el textgrid.
# WORDS: Palabra (no necesario)
# PHONES: Fonemas
# NLCE: Nasalancia del fonema
# RMS: RMS del intervalo del fonema
TEXTGRID_TIER = ['WORDS', 'PHONES', 'NLCE', 'RMS']


# Clases
CLASS_NAMES = ['CLASS_3', 'CLASS_5']

CLASS_3 = {
    'Nasal_Cons': 'Nasal',
    'Nasal_Vowel': 'Nasal',
    'Oral_Cons': 'Oral',
    'Oral_Vowel': 'Oral',
    'Silence': 'SL',
    'Default': ''
}

CLASS_5 = {
    'Nasal_Cons': 'NC',
    'Nasal_Vowel': 'NV',
    'Oral_Cons': 'OC',
    'Oral_Vowel': 'OV',
    'Silence': 'SL',
    'Default': ''
}

# Clasificación de los fonemas
VOCALES = ['a', 'e', 'i', 'o', 'u', 'j', 'w']
CONS_ORAL = ['b', 'c', 'd̪', 'f', 'k', 'l', 'p', 'r', 's', 'tʃ', 't̪', 'w', 'x', 
            'ç', 'ð', 'ɟ', 'ɟʝ', 'ɡ', 'ɣ', 'ɾ', 'ʃ', 'ʎ', 'ʝ', 'β', 'θ']
CONS_NASAL = ['ɲ', 'm', 'n', 'ŋ']

# Mínimo de nasalidad para cada una de las vocales
MIN_NASAL = {
    'a': 27,
    'e': 32,
    'i': 43,
    'o': 28,
    'u': 41,
    'j': 43,
    'w': 41,
    '': 0  # last is for default values   
}  

# Valor máximo de RMS para considerar un silencio
MAX_RMS_SL = 10


# Ajustes previos a la ejecución, se crean carpetas y se definen listas y paths
if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)

AUDIO_PATH = os.path.join(os.getcwd(), AUDIO_FOLDER, AUDIO_DATABASE)
RESULT_PATH = os.path.join(os.getcwd(), RESULT_FOLDER, RESULT_DATABASE)

MFCC_NAMES = [f"mfcc{x}" for x in range(0, NUMCEP)]
DELTA_NAMES = [f"delta{x}" for x in range(0, NUMCEP)]
DELTA_DELTA_NAMES = [f"deltaDelta{x}" for x in range(0, NUMCEP)]