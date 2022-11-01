from scipy.io import wavfile
import numpy as np
import librosa          # https://librosa.org/doc/latest/index.html
from pandas import DataFrame

import textgrid
from pathlib import Path
import pandas as pd 
import numpy as np
import os
import re

import csv

from scipy.io.wavfile import write
from scipy.io import wavfile 

from python_speech_features import mfcc     # https://python-speech-features.readthedocs.io/en/latest/
from python_speech_features import delta

import scipy.io.wavfile as wav


def check_wav_textgrid(audio_folder):
    ''' Comprueba que la carpeta audio_folder contiene archivos .wav y .textgrid en el mismo número.
        Devuelve dos listas con los path de los archivos, ordenados. 
    '''
    
    audio_list = [os.path.join(audio_folder,file) for file in os.listdir(audio_folder) if file.endswith('.wav')]
    textgrid_list = [os.path.join(audio_folder,file) for file in os.listdir(audio_folder) if file.endswith('.TextGrid')]

    if not audio_list: # check if empty
        print(f"No hay archivos .wav en la carpeta: {audio_folder}")
        print("Introduce lo archivos .wav y vuelve a ejecutar el scipt.")
        exit(1)
    elif not textgrid_list: # check if empty
        print(f"No hay archivos .TextGrid en la carpeta: {audio_folder}")
        print("Introduce lo archivos .TextGrid y vuelve a ejecutar el scipt.")
        exit(1)
    elif len(audio_list) != len(textgrid_list): 
        print(f"No hay el mismo número de archivos .wav que .TextGrid en la carpeta: {audio_folder}")
        print("Cada archivo .wav tiene que venir acompañado de su archivo .TextGrid. Introduce los archivos que falten y vuelve a ejecutar el script.")
        exit(1)
    else:
        print(f"Número de archivos .wav y .TextGrid correcto, continúa la ejecución del script.")

    return sorted(audio_list), sorted(textgrid_list)


def baseline_columns():
    ''' Comprueba que la base de datos existente tenga las mismas columnas que la nueva.
        Esto evita que se generen nuevos descriptores sobre una base de datos ya generada.
        La estructura de las columnas es:
            [SPEAKER  WINDOWS  FEATURES  TEXTGRID  CLASS]
    '''

    baseline_columns = COLUMNS_NAMES_SPEAKER + WIN_COLUMN_NAME

    if 'mfcc' in FEATURES_TO_CALCULATE:
        baseline_columns = baseline_columns + MFCC_NAMES

    if ('mfcc' in FEATURES_TO_CALCULATE) and ('delta' in FEATURES_TO_CALCULATE):
        baseline_columns = baseline_columns + DELTA_NAMES

    if ('mfcc' in FEATURES_TO_CALCULATE) and ('delta' in FEATURES_TO_CALCULATE) and ('deltaDelta' in FEATURES_TO_CALCULATE):
        baseline_columns = baseline_columns + DELTA_DELTA_NAMES

    baseline_columns = baseline_columns + TEXTGRID_TIER + CLASS_NAMES

    return baseline_columns  

def get_speakeraudio_file(audio_path):
    ''' Genera un dataframe con la información que se puede extraer del nombre del usuario. 
        La información tiene que estar separada por '_' y el nombre de cada parámetro viene definido en COLUMNS_NAMES_SPEAKER.
    '''

    speaker_list = (os.path.split(audio_path)[-1]).split('_') # (head,tail)
    speaker_list[-1] = speaker_list[-1].replace('.wav','')
    speaker_df = pd.DataFrame([speaker_list], columns = COLUMNS_NAMES_SPEAKER) 

    return speaker_df


def moving_window_textgrid(audio, width=WIN_LENGTH, overlap=WIN_OVERLAP):
    if type(audio) != list:
        audio = audio.tolist()
    return [max(set(audio[i : i + width]), key = audio[i : i + width].count) for i in range(0, len(audio), width-overlap)]


def compute_mfcc(audio_data, samplerate):
    ''' Calcula los mfcc en función de los parámetros definidos en config
    '''

    np_mfcc_feat = mfcc(signal=np.array(audio_data, dtype=np.single), samplerate=samplerate, 
                winlen=WIN_LENGTH*1e-3, winstep=(WIN_LENGTH-WIN_OVERLAP)*1e-3, numcep=NUM_MFCC, 
                nfilt=26, nfft=1024, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)

    df_mfcc_feat = pd.DataFrame(np_mfcc_feat, columns=MFCC_NAMES)

    return np_mfcc_feat, df_mfcc_feat


def compute_delta(input_data, num_delta=2, c_names=DELTA_NAMES):
    ''' Calcula los delta del mfcc en función de los parámetros definidos en config
    '''

    np_delta_feat = delta(input_data, num_delta)
    df_delta_feat = pd.DataFrame(np_delta_feat, columns=c_names)

    return np_delta_feat, df_delta_feat


def compute_textgrid(textgrid_file, samplerate, len_audio):

    print("Calculando Textgrid.")
    tg = textgrid.TextGrid.fromFile(textgrid_file)

    win_length_samples = int(WIN_LENGTH * samplerate / 1000) # ms
    win_overlap_samples = int(WIN_OVERLAP * samplerate / 1000) # ms

    txtgrid_speaker = []

    for interval_tier in tg:
        #print(f"Calculando IntervalTier {re.search('IntervalTier (.*), ', str(interval_tier)).group(1)}")
        tier_list = []

        for interval in interval_tier: # last annotation is invalid
            tier_list += [interval.mark]*int(np.round((interval.maxTime-interval.minTime)*samplerate))
        # end for

        win_tier_list = moving_window_textgrid(tier_list, win_length_samples, win_overlap_samples)
        txtgrid_speaker.append(win_tier_list[0:len_audio])
    # end for

    # Con la misma longitud y se convierten en las listas en columnas
    txtgrid_speaker = list(map(list, zip(*txtgrid_speaker)))
    #np_speaker = np.array(txtgrid_speaker)
    df_speaker = pd.DataFrame(txtgrid_speaker, columns=TEXTGRID_TIER)

    return df_speaker

def compute_class(df_textgrid):
	'''asd'''
	list_class_3 = []
	list_class_5 = []

	list_phones = list(df_textgrid['PHONES'])
	list_rms = list(df_textgrid['RMS'].astype(int))
	list_nasalance = list(df_textgrid['NLCE'].astype(int))

	for ph, rms, nlce in zip(list_phones, list_rms, list_nasalance):
		if (ph == '' and rms < 10):
			subclass = 'Silence'

		elif (ph == '' and rms > 10):
			subclass = 'Default'

		elif ph in VOCALES:
			if nlce > MIN_NASAL[ph]:
				subclass = 'Nasal_Vowel'
			else:
				subclass = 'Oral_Vowel'

		elif ph in CONS_ORAL:
			subclase = 'Oral_Cons'

		elif ph in CONS_NASAL:
			subclase = 'Nasal_Cons'
		
		else:
			#print(f"¡Aviso!: el símbolo {ph} anotado en el textGrid no está en \
			#	el diccionario. Se ignora.")
			subclass = 'Default'

		list_class_3.append(CLASS_3[subclass])
		list_class_5.append(CLASS_5[subclass])
		
	return pd.DataFrame(zip(list_class_3,list_class_5), columns = CLASS_NAMES)



## MAIN
if __name__ == "__main__":
    
    # Se comprueba que la carpeta AUDIO_PATH contiene archivos .wav y .textgrid
    audio_list, textgrid_list = check_wav_textgrid(AUDIO_PATH)


    # Se comprueba si existe la base de datos que se va a generar
    print(f"\nBase de datos a generar: {RESULT_PATH}")

    if os.path.exists(RESULT_PATH):
        print("Existe la base de datos\n")

        features_already_generated = pd.read_csv(RESULT_PATH)
        speakers_list = list(set(features_already_generated['speaker']))

        if (len(baseline_columns()) != len(features_already_generated.columns)) or (baseline_columns() != features_already_generated.columns):
            print("La base de datos existente no tiene los mismos descriptores. Cambia el nombre o borra el archivo original.")
            print("Saliendo del script.")
            exit(1)

    else:
        print("La base de datos NO existe. Se crea una nueva.\n")

        speakers_list = []
        features_already_generated = pd.DataFrame([])
    # end if

    # Se calculan los descriptores
    df_features_forloop = [] # descriptores calculados durante el bucle for


    for audio_file, textgrid_file in zip(audio_list,textgrid_list):

        list_df_features_speaker = [] # descriptores calculados para este speaker

        # Datos que se generan para cada audio 
        speaker_info_df = get_speakeraudio_file(audio_file)
        spk_loop = speaker_info_df["speaker"].values[0]

        # Comprueba si el speaker se encuentra en la base de datos
        if spk_loop in speakers_list:
            print(f"Locutor {spk_loop} ya se encuentra en la base de datos. Pasamos al siguiente.")
        else:
            print(f"Locutor {spk_loop} NUEVO. Se calculan los descriptores.")

            # Se lee el audio
            samplerate, audio_data = wavfile.read(audio_file)


            ## Primero se calculan los descriptores
            # MFCC
            if 'mfcc' in FEATURES_TO_CALCULATE:
                np_mfcc_feat, df_mfcc_feat = compute_mfcc(audio_data, samplerate)
                list_df_features_speaker.append(df_mfcc_feat)

            # Delta
            if ('mfcc' in FEATURES_TO_CALCULATE) and ('delta' in FEATURES_TO_CALCULATE):
                np_delta_feat, df_delta_feat = compute_delta(np_mfcc_feat, num_delta=NUM_DELTA, c_names=DELTA_NAMES)
                list_df_features_speaker.append(df_delta_feat)

            # Delta delta
            if ('mfcc' in FEATURES_TO_CALCULATE) and ('delta' in FEATURES_TO_CALCULATE) and ('deltaDelta' in FEATURES_TO_CALCULATE):
                np_delta_delta_feat, df_delta_delta_feat = compute_delta(np_delta_feat, num_delta=NUM_DELTA, c_names=DELTA_DELTA_NAMES)
                list_df_features_speaker.append(df_delta_delta_feat)

            # Completar resto de descriptores
            ##### POR COMPLETAR #####
            #
            #
            #
            ##### POR COMPLETAR #####


            ## Número de ventanas
            increasing_win_index = [f"w{str(x).zfill(NUM_ZEROS_WINDOWS_MAX)}" for x in range(0,len(list_df_features_speaker[0].index))]
            df_win_index = pd.DataFrame(increasing_win_index, columns=WIN_COLUMN_NAME)   
                

            ## Datos del locutor
            speaker_info_df = speaker_info_df.loc[speaker_info_df.index.repeat(len(list_df_features_speaker[0].index))].reset_index(drop=True)


            ## Textgrid
            df_textgrid = compute_textgrid(textgrid_file, samplerate, len(list_df_features_speaker[0].index))
            df_textgrid['NLCE'] = df_textgrid['NLCE'].replace('', '0')

            
            ## Clase de los fonemas
            df_class = compute_class(df_textgrid)


            # Se concatenan por columnas [A B C D]
            df_features_speaker = pd.concat([speaker_info_df, df_win_index] + list_df_features_speaker + [df_textgrid, df_class], axis=1)

            # Se concatena con lo que llevamos generado en el for loop
            if len(df_features_forloop):
                df_features_forloop = pd.concat([df_features_forloop, df_features_speaker])
            else:
                df_features_forloop = df_features_speaker
            # end if
        #end if
    # end for

    # Se concatenan con los datos previamente calculados en la base de datos
    if len(features_already_generated):
        print('Uniendo con la base de datos existente')
        df_features_final = pd.concat([features_already_generated, df_features_forloop])
    else:
        df_features_final = df_features_forloop


    # Se guarda como csv
    df_features_final.to_csv(RESULT_PATH, encoding='utf8', index=False)  

    print(f"\nFIN calculo descriptores") 