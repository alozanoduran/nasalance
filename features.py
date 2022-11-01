"""Script principal para el cálculo de descriptores de audio.

Este script lee una serie de de archivos .wav y .TextGrid de una carpeta de 
entrada y crea un fichero de salida .csv con los siguientes resultados:

    - Datos del locutor del audio.
    - Numeración de las ventanas de audio.
    - Descriptores calculados.
    - Parámetros extraidos del archivo textgrid.
    - Clase de los fonemas.

Si la base de datos ya existe se añaden los nuevos locutores al archivo 
existente.

Los parámetros de configuración del script se encuentra en features_config.py.
"""

import os

import numpy as np
import pandas as pd
from scipy.io import wavfile

from python_speech_features import delta
from python_speech_features import mfcc  
import textgrid

import features_config as cfg


def check_wav_textgrid(audio_folder):
    """Comprueba la carpeta audio_folder.

    Comprueba que la carpeta audio_folder contenga el mismo número de archivos 
    .wav y .textgrid.

    Args:
      audio_folder:
        Path a la carpeta de audios que se va a procesar.

    Returns:
      Dos listas ordenadas con los path completos de los ficheros .wav y 
      .textgrid.
    """

    audio_list = [
        os.path.join(audio_folder, file)
        for file in os.listdir(audio_folder)
        if file.endswith('.wav')
    ]

    textgrid_list = [
        os.path.join(audio_folder, file)
        for file in os.listdir(audio_folder)
        if file.endswith('.TextGrid')
    ]

    if not audio_list:  # check if empty
        print(f"No hay archivos .wav en la carpeta: {audio_folder}")
        print("Introduce lo archivos .wav y vuelve a ejecutar el scipt.")
        exit(1)
    elif not textgrid_list:  # check if empty
        print(f"No hay archivos .TextGrid en la carpeta: {audio_folder}")
        print("Introduce lo archivos .TextGrid y vuelve a ejecutar el scipt.")
        exit(1)
    elif len(audio_list) != len(textgrid_list):
        print(f"No hay el mismo número de archivos .wav que .TextGrid en la \
            carpeta: {audio_folder}")
        print("Cada archivo .wav tiene que venir acompañado de su archivo \
            .TextGrid. Introduce los archivos que falten y vuelve a ejecutar \
            el script.")
        exit(1)
    else:
        print(f"Número de archivos .wav y .TextGrid correcto, continúa la \
            ejecución del script.")

    return sorted(audio_list), sorted(textgrid_list)


def baseline_columns():
    """Genera los nombre de columnas para la base de datos de salida.

    Esto evita que se añadan nuevos descriptores sobre una base de datos ya 
    generada.

    La estructura de las columnas es:
        [SPEAKER  WINDOWS  FEATURES  TEXTGRID  CLASS]

    Returns:
      Una lista con los nombres de las columnas de la base de datos.
    """

    baseline_columns = cfg.COLUMNS_NAMES_SPEAKER + cfg.WIN_COLUMN_NAME

    if 'mfcc' in cfg.FEATURES_TO_CALCULATE:
        baseline_columns = baseline_columns + cfg.MFCC_NAMES

    if ('mfcc' in cfg.FEATURES_TO_CALCULATE 
        and 'delta' in cfg.FEATURES_TO_CALCULATE):
        baseline_columns = baseline_columns + cfg.DELTA_NAMES

    if ('mfcc' in cfg.FEATURES_TO_CALCULATE 
        and 'delta' in cfg.FEATURES_TO_CALCULATE 
        and 'deltaDelta' in cfg.FEATURES_TO_CALCULATE):
        baseline_columns = baseline_columns + cfg.DELTA_DELTA_NAMES

    baseline_columns = baseline_columns + cfg.TEXTGRID_TIER + cfg.CLASS_NAMES

    return baseline_columns


def get_speakeraudio_file(audio_path):
    """Genera un dataframe con la información del usuario.

    La información del usuario se extrae del nombre del fichero de audio, con 
    los campos separados por '_'. 

    Args:
      audio_path:
        Path al fichero de audio.

    Returns:
      Dataframe con los datos del usuario.
    """

    speaker_list = (os.path.split(audio_path)[-1]).split('_')  # (head,tail)
    speaker_list[-1] = speaker_list[-1].replace('.wav', '')

    return pd.DataFrame([speaker_list], columns=cfg.COLUMNS_NAMES_SPEAKER)


def moving_window_textgrid(data, width=cfg.WIN_LENGTH, overlap=cfg.WIN_OVERLAP):
    """Genera una ventana móvil y devuelve el dato más repetido en la ventana.

    Args:
      data:
        Lista de entrada.
      width:
        Longitud de la ventana.
      overlap:
        Superposición de la ventana.

    Returns:
      Lista con el dato más repetido para cada una de las ventanas.
    """
    
    if type(data) != list:
        data = data.tolist()
    return [
        max(set(data[i:i + width]), key=data[i:i + width].count)
        for i in range(0, len(data), width - overlap)
    ]


def compute_mfcc(audio_data, samplerate):
    """Calcula los parámetros mfcc.

    Args:
      audio_data:
        Datos de entrada.
      samplerate:
        Frecuencia de muestreo en Hz.

    Returns:
      Array con los mfcc calculados.
      Dataframe con los mfcc calculados.
    """

    np_mfcc_feat = mfcc(signal=np.array(audio_data, dtype=np.single),
                        samplerate=samplerate,
                        winlen=cfg.WIN_LENGTH * 1e-3,
                        winstep=(cfg.WIN_LENGTH - cfg.WIN_OVERLAP) * 1e-3,
                        numcep=cfg.NUMCEP,
                        nfilt=cfg.NFILT,
                        nfft=cfg.NFFT,
                        lowfreq=cfg.LOWFREQ,
                        highfreq=cfg.HIGHFREQ,
                        preemph=cfg.PREEMPH,
                        ceplifter=cfg.CEPLIFTER,
                        appendEnergy=cfg.APPENDENERGY,
                        winfunc=np.hamming)

    df_mfcc_feat = pd.DataFrame(np_mfcc_feat, columns=cfg.MFCC_NAMES)

    return np_mfcc_feat, df_mfcc_feat


def compute_delta(input_data, num_delta=2, c_names=cfg.DELTA_NAMES):
    """Calcula los parámetros delta aplicados al mfcc.

    Args:
      audio_data:
        Datos de entrada.
      samplerate:
        Frecuencia de muestreo en Hz.
      c_names:
        Nombre de las columnas. La misma función se usa para delta y deltaDelta.

    Returns:
      Array con los mfcc calculados.
      Dataframe con los mfcc calculados.
    """

    np_delta_feat = delta(input_data, num_delta)
    df_delta_feat = pd.DataFrame(np_delta_feat, columns=c_names)

    return np_delta_feat, df_delta_feat


def compute_textgrid(textgrid_file, samplerate, len_audio):
    """Extrae los parámetro del archivo TextGrid en función de los tiers.

    Args:
      textgrid_file:
        Path del archivo TextGrid.
      samplerate:
        Frecuencia de muestreo en Hz.
      len_audio:
        Longitud del audio. Permite conocer el fin de las anotaciones.

    Returns:
      Dataframe con los tiers definidos en el fichero Textgrid.
    """
    print("Calculando Textgrid.")
    tg = textgrid.TextGrid.fromFile(textgrid_file)

    win_length_samples = int(cfg.WIN_LENGTH * samplerate / 1000)  # ms
    win_overlap_samples = int(cfg.WIN_OVERLAP * samplerate / 1000)  # ms

    txtgrid_speaker = []

    for interval_tier in tg:
        tier_list = []

        for interval in interval_tier:  
            tier_list += [interval.mark] * int(
                np.round((interval.maxTime - interval.minTime) * samplerate))
        # end for

        win_tier_list = moving_window_textgrid(tier_list, win_length_samples,
                                               win_overlap_samples)
        txtgrid_speaker.append(win_tier_list[0:len_audio])
        # Pueden existir anotaciones fuera del audio, con esto se evita.

    # Se convierte las listas en columnas
    txtgrid_speaker = list(map(list, zip(*txtgrid_speaker)))
    return pd.DataFrame(txtgrid_speaker, columns=cfg.TEXTGRID_TIER)


def compute_class(df_textgrid):
    """Calcula la clase del fonema.

    Args:
      df_textgrid:
        Dataframe generado mediante la función compute_textgrid.

    Returns:
      Dataframe con la clase (3 y 5) asociada a cada fonema.
    """
    list_class_3 = []
    list_class_5 = []

    list_phones = list(df_textgrid['PHONES'])
    list_rms = list(df_textgrid['RMS'].astype(float))
    list_nasalance = list(df_textgrid['NLCE'].astype(float))

    for ph, rms, nlce in zip(list_phones, list_rms, list_nasalance):
        if (ph == '' and rms < 10):
            subclass = 'Silence'
        elif (ph == '' and rms > 10):
            subclass = 'Default'
        elif ph in cfg.VOCALES:
            if nlce > cfg.MIN_NASAL[ph]:
                subclass = 'Nasal_Vowel'
            else:
                subclass = 'Oral_Vowel'
        elif ph in cfg.CONS_ORAL:
            subclass = 'Oral_Cons'
        elif ph in cfg.CONS_NASAL:
            subclass = 'Nasal_Cons'
        else:
            subclass = 'Default'

        list_class_3.append(cfg.CLASS_3[subclass])
        list_class_5.append(cfg.CLASS_5[subclass])

    return pd.DataFrame(zip(list_class_3, list_class_5), columns=cfg.CLASS_NAMES)


def compute_database(audio_list, textgrid_list, speakers_list):
    """Calcula todos los descriptores y datos necesarios para generar la base de
    datos.

    Args:
      audio_list:
        Lista con los path de los archivos de audio.
      textgrid_list:
        Lista con los path de los archivos TextGrid.
      speakers_list:
        Lista con los locutores que existen en la base de datos ya generada. 
        Puede estar vacío cuando se genera la primera vez.

    Returns:
      Dataframe con todos los descriptores calculados en el bucle for.
    """
    df_feat_forloop = []  # descriptores calculados durante el bucle for

    for audio_file, textgrid_file in zip(audio_list, textgrid_list):

        list_df_feat_spk = []  # descriptores calculados speaker

        # Datos que se generan para cada audio
        speaker_info_df = get_speakeraudio_file(audio_file)
        spk_loop = speaker_info_df["speaker"].values[0]

        # Comprueba si el speaker se encuentra en la base de datos
        if spk_loop in speakers_list:
            print(f"Locutor {spk_loop} ya se encuentra en la base de datos.")
        else:
            print(f"Locutor {spk_loop} NUEVO. Se calculan los descriptores.")

            # Se lee el audio
            samplerate, audio_data = wavfile.read(audio_file)

            ## Primero se calculan los descriptores
            list_df_feat_spk = compute_features_speaker(
                                                        audio_data, 
                                                        samplerate)

            ## Número de ventanas
            increasing_win_index = [
                f"w{str(x).zfill(cfg.NUM_ZEROS_WINDOWS_MAX)}"
                for x in range(0, len(list_df_feat_spk[0].index))
            ]
            df_win_index = pd.DataFrame(increasing_win_index,
                                        columns=cfg.WIN_COLUMN_NAME)

            ## Datos del locutor
            speaker_info_df = speaker_info_df.loc[speaker_info_df.index.repeat(
                len(list_df_feat_spk[0].index))].reset_index(drop=True)

            ## Textgrid
            df_textgrid = compute_textgrid(
                textgrid_file, samplerate,
                len(list_df_feat_spk[0].index))
            df_textgrid['NLCE'] = df_textgrid['NLCE'].replace('', '0')

            ## Clase de los fonemas
            df_class = compute_class(df_textgrid)

            # Se concatenan por columnas [A B C D]
            df_features_speaker = pd.concat([speaker_info_df, df_win_index] +
                                            list_df_feat_spk +
                                            [df_textgrid, df_class],
                                            axis=1)

            # Se concatena con lo que llevamos generado en el for loop
            if len(df_feat_forloop):
                df_feat_forloop = pd.concat(
                    [df_feat_forloop, df_features_speaker])
            else:
                df_feat_forloop = df_features_speaker
            # end if
        #end if
    # end for
    return df_feat_forloop


def compute_features_speaker(audio_data, samplerate):
    """Calcula todos los descriptores y datos necesarios para generar la base de
    datos.

    Args:
      audio_data:
        Audio en forma de numpy array.
      samplerate:
        Frecuencia de muestreo del audio.

    Returns:
      Lista con los dataframe calculados para todos los descriptores.
    """

    list_df_features_speaker = []  

    # MFCC
    if 'mfcc' in cfg.FEATURES_TO_CALCULATE:
        np_mfcc_feat, df_mfcc_feat = compute_mfcc(audio_data, samplerate)
        list_df_features_speaker.append(df_mfcc_feat)

    # Delta
    if ('mfcc' in cfg.FEATURES_TO_CALCULATE
        ) and ('delta' in cfg.FEATURES_TO_CALCULATE):
        np_delta_feat, df_delta_feat = compute_delta(
                                    np_mfcc_feat, 
                                    num_delta=cfg.DIST_DELTA, 
                                    c_names=cfg.DELTA_NAMES)
        list_df_features_speaker.append(df_delta_feat)

    # Delta delta
    if ('mfcc' in cfg.FEATURES_TO_CALCULATE
        ) and ( 'delta' in cfg.FEATURES_TO_CALCULATE
        ) and ('deltaDelta' in cfg.FEATURES_TO_CALCULATE):
        np_delta_delta_feat, df_delta_delta_feat = compute_delta(
                                                np_delta_feat,
                                                num_delta=cfg.DIST_DELTA,
                                                c_names=cfg.DELTA_DELTA_NAMES)
        list_df_features_speaker.append(df_delta_delta_feat)

    # Completar resto de descriptores
    # TODO

    return list_df_features_speaker



## MAIN
if __name__ == "__main__":

    # Se comprueba que la carpeta AUDIO_PATH contiene archivos .wav y .textgrid
    audio_list, textgrid_list = check_wav_textgrid(cfg.AUDIO_PATH)

    # Se comprueba si existe la base de datos que se va a generar
    print(f"\nBase de datos a generar: {cfg.RESULT_PATH}")

    if os.path.exists(cfg.RESULT_PATH):
        print("Existe la base de datos\n")

        feat_already_generated = pd.read_csv(cfg.RESULT_PATH)
        speakers_list = list(set(feat_already_generated['speaker']))

        if (len(baseline_columns()) != len(feat_already_generated.columns)
           ) or (baseline_columns() != feat_already_generated.columns).all():
            print("La base de datos existente no tiene los mismos descriptores.") 
            print("Cambia el nombre o borra el archivo original.")
            print("Saliendo del script.")
            exit(1)
    else:
        print("La base de datos NO existe. Se crea una nueva.\n")

        speakers_list = []
        feat_already_generated = pd.DataFrame([])
    # end if

    # Se calculan los descriptores
    df_feat_forloop = compute_database(audio_list, textgrid_list, speakers_list)

    # Se concatenan con los datos previamente calculados en la base de datos
    if len(feat_already_generated) and len(df_feat_forloop):
        print('Uniendo con la base de datos existente.')
        df_features_final = pd.concat(
            [feat_already_generated, df_feat_forloop])
    else:
        df_features_final = df_feat_forloop

    # Se guarda como csv
    if len(df_features_final):
        df_features_final.to_csv(cfg.RESULT_PATH, encoding='utf8', index=False)
        print(f"\nFIN calculo descriptores.")
    else:
        print(f"\nFIN calculo descriptores. No se ha realizado ningún cálculo")