#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on Tue May 23 16:13:48 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.5'
expName = 'MemoryTask'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/raphaelgabiazon/Desktop/fMRI_Pilot_Jitter_Test_1/fMRI_MemoryTask_Pilot_Jitter_Test_01-Builder.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1470, 956], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "setUp" ---
# Run 'Begin Experiment' code from setUpCode
import random

# Function to save data in three-column format
def save_three_column_file(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            f.write('{:.1f}\t{:.3f}\t{:.3f}\n'.format(row[0], row[1], row[2]))

# Function generate random jitter times
def generate_jitter_time(min_: float, max_: float, count: int, target: float, epsilon=1e-10) -> list:
    # Make sure the target is reachable
    if target < min_ * count or target > max_ * count:
        raise ValueError(f"Target value '{target}' is not valid for allowed range '{min_}' - '{max_}'")

    numbers = []
    
    # Execute the next steps `count` times
    for i in range(count):
        # Generate a random float number within the allowed range
        number = random.uniform(min_, max_ + epsilon)
        
        current_sum = sum(numbers) + number
        delta = target - current_sum
        remaining_tries = count - i - 1
        missing = 0
        
        # The last number is what's remaining
        if remaining_tries == 0:
            number = target - sum(numbers)
        
        # Ensure that the target is still reachable
        elif delta > remaining_tries * max_:
            min_to_add = delta - remaining_tries * max_
            missing = random.uniform(min_to_add, max_ - number)
            
        elif delta < remaining_tries * min_:
            min_to_remove = remaining_tries * min_ - delta
            missing = -random.uniform(min_to_remove, number - min_)
            
        # Correct the number if it needs to be
        number += missing

        # Round the number to one decimal place
        number = round(number, 1)

        numbers.append(number)
        
    return numbers

# Generate 11 numbers between 5 and 7 that sum up to 66.0 
jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)

# Initialize variables
volume_counter = 1
volume_total = 672 # (10.0 sec + 66.0 sec + 36.0 sec) x 6 conditions

start_time = 0
current_time = 0

# Create clocks to keep track of the time
timer = core.Clock()

# Create a global clock 
Time_Since_Run = core.Clock()

cross_time = 6.0 # Fixation cross duration
image_time = 3.0 # Stimulus duration
key_board_time = 3.0 # Keyboard response duration during trial

# Initalize fixation cross start time, SOA after block 1 depend on this
next_fixCross_started_time = 0  

Block_1_Run = 1
Block_2_Run = 2
Block_3_Run = 3

# Create counters 
block_loop_trials_counter = 0 # counter for trials in block loops
encode_trials_counter = 0 # counter for encode trials
recognition_trials_counter = 0 # counter for recognition trials
Block_Loop_Voulme_Counter = [] # To reset volume counter

# Initialize the current encoding, recognition, and jitter indices for tracking the current image position and jitter time
current_encode_index = 0
current_recog_index = 0
current_jitter_index = 0

# Initialize the current encoding key index for tracking the list condition in Encoding_Dictionary_Block_1_Run_1
current_encode_key_index = 0

# Set flags for instruction display based on block and condition
Block_1_Run_1_Started = True # Assuming Block 1 Run 1 is first
Block_2_Run_2_Started = False
Block_3_Run_3_Started = False
Fix_Cross_Started = False # Becomes true after first stimulus 
Encoding_Started = True # Encoding condition always first
Recognition_Started = False

# --- Initialize components for Routine "condLoader" ---
# Run 'Begin Experiment' code from condLoaderCod
import pandas as pd
import random 

#BLOCK 1 RUN 1
# Load encoding and recognition conditions and store in dataframe (df)
Block_1_Run_1_Conditions_file = 'conditions.xlsx'
Block_1_Run_1_Conditions_df = pd.read_excel(Block_1_Run_1_Conditions_file)

# Load encoding conditions
Face_Encoding_Block_1_Run_1_file = Block_1_Run_1_Conditions_df.loc[0, 'cond_file']
Place_Encoding_Block_1_Run_1_file  = Block_1_Run_1_Conditions_df.loc[1, 'cond_file']
Pair_Encoding_Block_1_Run_1_file= Block_1_Run_1_Conditions_df.loc[2, 'cond_file']

# Load recognition conditions
Face_Recog_Block_1_Run_1_file = Block_1_Run_1_Conditions_df.loc[3, 'cond_file']
Place_Recog_Block_1_Run_1_file = Block_1_Run_1_Conditions_df.loc[4, 'cond_file']
Pair_Recog_Block_1_Run_1_file = Block_1_Run_1_Conditions_df.loc[5, 'cond_file']

# Store encoding conditions in dataframe (df)
Face_Encoding_Stimuli_Block_1_Run_1_df = pd.read_excel(Face_Encoding_Block_1_Run_1_file)
Place_Encoding_Stimuli_Block_1_Run_1_df = pd.read_excel(Place_Encoding_Block_1_Run_1_file)
Pair_Encoding_Stimuli_Block_1_Run_1_df = pd.read_excel(Pair_Encoding_Block_1_Run_1_file)

# Store recognition conditions in dataframe (df)
Face_Recog_Stimuli_Block_1_Run_1_df = pd.read_excel(Face_Recog_Block_1_Run_1_file)
Place_Recog_Stimuli_Block_1_Run_1_df = pd.read_excel(Place_Recog_Block_1_Run_1_file)
Pair_Recog_Stimuli_Block_1_Run_1_df = pd.read_excel(Pair_Recog_Block_1_Run_1_file)

# Create encoding list of images conditions
Face_Encoding_Images_Block_1_Run_1 = Face_Encoding_Stimuli_Block_1_Run_1_df["encoding_stims"].tolist()
Place_Encoding_Images_Block_1_Run_1 = Place_Encoding_Stimuli_Block_1_Run_1_df["encoding_stims"].tolist()
Pair_Encoding_Images_Block_1_Run_1 = Pair_Encoding_Stimuli_Block_1_Run_1_df["encoding_stims"].tolist()

# Create recognition list of images conditions
Face_Recog_Images_Block_1_Run_1 = Face_Recog_Stimuli_Block_1_Run_1_df["recog_stims"].tolist()
Place_Recog_Images_Block_1_Run_1 = Place_Recog_Stimuli_Block_1_Run_1_df["recog_stims"].tolist()
Pair_Recog_Images_Block_1_Run_1 = Pair_Recog_Stimuli_Block_1_Run_1_df["recog_stims"].tolist()

# BLOCK 2 RUN 2
# Load encoding and recognition conditions and store in dataframe (df)
Block_2_Run_2_Conditions_file = 'conditions.xlsx'
Block_2_Run_2_Conditions_df = pd.read_excel(Block_2_Run_2_Conditions_file)

# Load encoding conditions
Face_Encoding_Block_2_Run_2_file = Block_2_Run_2_Conditions_df.loc[6, 'cond_file']
Place_Encoding_Block_2_Run_2_file  = Block_2_Run_2_Conditions_df.loc[7, 'cond_file']
Pair_Encoding_Block_2_Run_2_file = Block_2_Run_2_Conditions_df.loc[8, 'cond_file']

# Load recognition conditions
Face_Recog_Block_2_Run_2_file = Block_2_Run_2_Conditions_df.loc[9, 'cond_file']
Place_Recog_Block_2_Run_2_file = Block_2_Run_2_Conditions_df.loc[10, 'cond_file']
Pair_Recog_Block_2_Run_2_file = Block_2_Run_2_Conditions_df.loc[11, 'cond_file']

# Store encoding conditions in dataframe (df)
Face_Encoding_Stimuli_Block_2_Run_2_df = pd.read_excel(Face_Encoding_Block_2_Run_2_file)
Place_Encoding_Stimuli_Block_2_Run_2_df = pd.read_excel(Place_Encoding_Block_2_Run_2_file)
Pair_Encoding_Stimuli_Block_2_Run_2_df = pd.read_excel(Pair_Encoding_Block_2_Run_2_file)

# Store recognition conditions in dataframe (df)
Face_Recog_Stimuli_Block_2_Run_2_df = pd.read_excel(Face_Recog_Block_2_Run_2_file)
Place_Recog_Stimuli_Block_2_Run_2_df = pd.read_excel(Place_Recog_Block_2_Run_2_file)
Pair_Recog_Stimuli_Block_2_Run_2_df = pd.read_excel(Pair_Recog_Block_2_Run_2_file)

# Create encoding list of images conditions
Face_Encoding_Images_Block_2_Run_2 = Face_Encoding_Stimuli_Block_2_Run_2_df["encoding_stims"].tolist()
Place_Encoding_Images_Block_2_Run_2 = Place_Encoding_Stimuli_Block_2_Run_2_df["encoding_stims"].tolist()
Pair_Encoding_Images_Block_2_Run_2 = Pair_Encoding_Stimuli_Block_2_Run_2_df["encoding_stims"].tolist()

# Create recognition list of images conditions
Face_Recog_Images_Block_2_Run_2 = Face_Recog_Stimuli_Block_2_Run_2_df["recog_stims"].tolist() 
Place_Recog_Images_Block_2_Run_2 = Place_Recog_Stimuli_Block_2_Run_2_df["recog_stims"].tolist()
Pair_Recog_Images_Block_2_Run_2 = Pair_Recog_Stimuli_Block_2_Run_2_df["recog_stims"].tolist()

# BLOCK 3 RUN 3
# Load encoding and recognition conditions and store in dataframe (df)
Block_3_Run_3_Conditions_file = 'conditions.xlsx'
Block_3_Run_3_Conditions_df = pd.read_excel(Block_3_Run_3_Conditions_file)

# Load encoding conditions
Face_Encoding_Block_3_Run_3_file = Block_3_Run_3_Conditions_df.loc[12, 'cond_file']
Place_Encoding_Block_3_Run_3_file  = Block_3_Run_3_Conditions_df.loc[13, 'cond_file']
Pair_Encoding_Block_3_Run_3_file = Block_3_Run_3_Conditions_df.loc[14, 'cond_file']

# Load recognition conditions
Face_Recog_Block_3_Run_3_file = Block_3_Run_3_Conditions_df.loc[15, 'cond_file']
Place_Recog_Block_3_Run_3_file = Block_3_Run_3_Conditions_df.loc[16, 'cond_file']
Pair_Recog_Block_3_Run_3_file = Block_3_Run_3_Conditions_df.loc[17, 'cond_file']

# Store encoding conditions in dataframe (df)
Face_Encoding_Stimuli_Block_3_Run_3_df = pd.read_excel(Face_Encoding_Block_3_Run_3_file)
Place_Encoding_Stimuli_Block_3_Run_3_df = pd.read_excel(Place_Encoding_Block_3_Run_3_file)
Pair_Encoding_Stimuli_Block_3_Run_3_df = pd.read_excel(Pair_Encoding_Block_3_Run_3_file)

# Store recognition conditions in dataframe (df)
Face_Recog_Stimuli_Block_3_Run_3_df = pd.read_excel(Face_Recog_Block_3_Run_3_file)
Place_Recog_Stimuli_Block_3_Run_3_df = pd.read_excel(Place_Recog_Block_3_Run_3_file)
Pair_Recog_Stimuli_Block_3_Run_3_df = pd.read_excel(Pair_Recog_Block_3_Run_3_file)

# Create encoding list of images conditions
Face_Encoding_Images_Block_3_Run_3 = Face_Encoding_Stimuli_Block_3_Run_3_df["encoding_stims"].tolist()
Place_Encoding_Images_Block_3_Run_3 = Place_Encoding_Stimuli_Block_3_Run_3_df["encoding_stims"].tolist()
Pair_Encoding_Images_Block_3_Run_3 = Pair_Encoding_Stimuli_Block_3_Run_3_df["encoding_stims"].tolist()

# Create recognition list of images conditions
Face_Recog_Images_Block_3_Run_3 = Face_Recog_Stimuli_Block_3_Run_3_df["recog_stims"].tolist() 
Place_Recog_Images_Block_3_Run_3 = Place_Recog_Stimuli_Block_3_Run_3_df["recog_stims"].tolist()
Pair_Recog_Images_Block_3_Run_3 = Pair_Recog_Stimuli_Block_3_Run_3_df["recog_stims"].tolist()

# Randomize each list
# BLOCK 1 RUN 1
random.shuffle(Face_Encoding_Images_Block_1_Run_1)
random.shuffle(Place_Encoding_Images_Block_1_Run_1)
random.shuffle(Pair_Encoding_Images_Block_1_Run_1)

random.shuffle(Face_Recog_Images_Block_1_Run_1)
random.shuffle(Place_Recog_Images_Block_1_Run_1)
random.shuffle(Pair_Recog_Images_Block_1_Run_1)

# BLOCK 2 RUN 2
random.shuffle(Face_Encoding_Images_Block_2_Run_2)
random.shuffle(Place_Encoding_Images_Block_2_Run_2)
random.shuffle(Pair_Encoding_Images_Block_2_Run_2)

random.shuffle(Face_Recog_Images_Block_2_Run_2)
random.shuffle(Place_Recog_Images_Block_2_Run_2)
random.shuffle(Pair_Recog_Images_Block_2_Run_2)

# BLOCK 3 RUN 3
random.shuffle(Face_Encoding_Images_Block_3_Run_3)
random.shuffle(Place_Encoding_Images_Block_3_Run_3)
random.shuffle(Pair_Encoding_Images_Block_3_Run_3)

random.shuffle(Face_Recog_Images_Block_3_Run_3)
random.shuffle(Place_Recog_Images_Block_3_Run_3)
random.shuffle(Pair_Recog_Images_Block_3_Run_3)

# BLOCK 1 RUN 1 DICTIONARIES
# Create a dictionary for encoding with keys for face, place, and pair encoding images
# and their respective randomized lists as values for Block 1 Run 1
Encoding_Dictionary_Block_1_Run_1 = {
    'Face_Encode_Key_Block_1_Run_1': Face_Encoding_Images_Block_1_Run_1,
    'Place_Encode_Key_Block_1_Run_1': Place_Encoding_Images_Block_1_Run_1,
    'Pair_Encode_Key_Block_1_Run_1': Pair_Encoding_Images_Block_1_Run_1
}

# Create a dictionary for recognition with keys for face, place, and pair recognition images
# and their respective randomized lists as values for Block 1 Run 1
Recognition_Dictionary_Block_1_Run_1 = {
    'Face_Recog_Key_Block_1_Run_1': Face_Recog_Images_Block_1_Run_1,
    'Place_Recog_Key_Block_1_Run_1': Place_Recog_Images_Block_1_Run_1,
    'Pair_Recog_Key_Block_1_Run_1': Pair_Recog_Images_Block_1_Run_1
}

# BLOCK 2 RUN 2 DICTIONARIES
# Create a dictionary for encoding with keys for face, place, and pair encoding images
# and their respective randomized lists as values for Block 2 Run 2
Encoding_Dictionary_Block_2_Run_2 = {
    'Face_Encode_Key_Block_2_Run_2': Face_Encoding_Images_Block_2_Run_2,
    'Place_Encode_Key_Block_2_Run_2': Place_Encoding_Images_Block_2_Run_2,
    'Pair_Encode_Key_Block_2_Run_2': Pair_Encoding_Images_Block_2_Run_2
}

# Create a dictionary for recognition with keys for face, place, and pair recognition images
# and their respective randomized lists as values for Block 2 Run 2
Recognition_Dictionary_Block_2_Run_2 = {
    'Face_Recog_Key_Block_2_Run_2': Face_Recog_Images_Block_2_Run_2,
    'Place_Recog_Key_Block_2_Run_2': Place_Recog_Images_Block_2_Run_2,
    'Pair_Recog_Key_Block_2_Run_2': Pair_Recog_Images_Block_2_Run_2
}

# BLOCK 3 RUN 3 DICTIONARIES
# Create a dictionary for encoding with keys for face, place, and pair encoding images
# and their respective randomized lists as values for Block 3 Run 3
Encoding_Dictionary_Block_3_Run_3 = {
    'Face_Encode_Key_Block_3_Run_3': Face_Encoding_Images_Block_3_Run_3,
    'Place_Encode_Key_Block_3_Run_3': Place_Encoding_Images_Block_3_Run_3,
    'Pair_Encode_Key_Block_3_Run_3': Pair_Encoding_Images_Block_3_Run_3
}

# Create a dictionary for recognition with keys for face, place, and pair recognition images
# and their respective randomized lists as values for Block 3 Run 3
Recognition_Dictionary_Block_3_Run_3 = {
    'Face_Recog_Key_Block_3_Run_3': Face_Recog_Images_Block_3_Run_3,
    'Place_Recog_Key_Block_3_Run_3': Place_Recog_Images_Block_3_Run_3,
    'Pair_Recog_Key_Block_3_Run_3': Pair_Recog_Images_Block_3_Run_3
}


# --- Initialize components for Routine "triggerSync" ---
triggerSyncText = visual.TextStim(win=win, name='triggerSyncText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
triggerSync_Key_Resp = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "encodeTrial_Run1" ---
# Run 'Begin Experiment' code from encodeTrialRun1Code
from psychopy.hardware import keyboard
from psychopy import core

# Function to store encode condition status and subject response
def update_encode_pre_recog_response_data(block_num, run_num, current_displayed_image, random_encode_key, response_key, liquid_conditions):
    gender_is_correct = None
    liquid_is_correct = None
    response_pair_fit = None

    image_gender = None
    response_gender = None
    image_water = None
    response_water = None
    response_pair_fit = None

    # Face encode condition
    if random_encode_key == f'Face_Encode_Key_Block_{block_num}_Run_{run_num}':
         image_gender = 'male' if 'om' in current_displayed_image or 'ym' in current_displayed_image else 'female'
         
         if response_key == '1':
            response_gender = 'male'
         elif response_key == '2':
            response_gender = 'female'
         else:
            response_gender = 'none'
         
         gender_is_correct = 'NO RESPONSE' if response_gender == 'none' else response_gender == image_gender
    
    # Place encode condition
    elif random_encode_key == f'Place_Encode_Key_Block_{block_num}_Run_{run_num}':
        image_water = 'no_liquid'
        for substr in liquid_conditions:
            if substr in current_displayed_image:
                image_water = 'liquid'
                break
    
        if response_key == '1':
            response_water = 'liquid'
        elif response_key == '2':
            response_water = 'no_liquid'
        else:
            response_water = 'none'
   
        liquid_is_correct = 'NO RESPONSE' if response_water == 'none' else response_water == image_water
    
    # Pair encode condition
    elif random_encode_key == f'Pair_Encode_Key_Block_{block_num}_Run_{run_num}':
        
        if response_key == '1':
            response_pair_fit = 'fits'
        elif response_key == '2':
            response_pair_fit = 'does_not_fit'
        else:
            response_pair_fit = 'none'

    return gender_is_correct, liquid_is_correct, response_pair_fit, image_gender, response_gender, image_water, response_water, response_pair_fit

kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Initialize the encoding lists to store encoding txt data
Encoding_Data_Face_Block_1_Run_1 = []
Encoding_Data_Place_Block_1_Run_1 = []
Encoding_Data_Pair_Block_1_Run_1 = []

# Initialize the encoding response data as dictionaries
Encoding_Response_Data_Face_Block_1_Run_1 = {}
Encoding_Response_Data_Place_Block_1_Run_1 = {}
Encoding_Response_Data_Pair_Block_1_Run_1 = {}

# Initialize the image_start_time variable
image_start_time = 0



encodeTrialRun1_Image = visual.ImageStim(
    win=win,
    name='encodeTrialRun1_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
encodeTrialTestKeyRespRun1 = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "recognitionTrial_Run1" ---
# Run 'Begin Experiment' code from recognitonTrialRun1Code
from psychopy.hardware import keyboard
from psychopy import core

# Function to store recognition response for d-prime
def d_prime(image_old_new, response_key):
    is_correct = None

    # "OLD" images condition
    if image_old_new == "OLD":
        if response_key == '1':
            is_correct = 'HIT'
        elif response_key == '2':
            is_correct = 'MISS'
        else:
            is_correct = 'NO RESPONSE'

    # "NEW" images condition
    elif image_old_new == "NEW":
        if response_key == '1':
            is_correct = 'FALSE ALARM'
        elif response_key == '2':
            is_correct = 'CORRECT REJECTION'
        else:
            is_correct = "NO RESPONSE"

    return is_correct

# Function to update encoding and recognition txt lists based on d - prime
def update_encoding_recognition_response_data( 
    block_num,  
    run_num, 
    is_correct, 
    current_displayed_image, 
    recog_key, 
    image_start_time, 
    image_time, 
    recog_response_data_face, 
    recog_response_data_place, 
    recog_response_data_pair, 
    encoding_response_data_face, 
    encoding_response_data_place, 
    encoding_response_data_pair
):

    # If the response is correct
    if is_correct == 'HIT' or is_correct == 'CORRECT REJECTION':
        # Add to txt only if responses are correct 
        if recog_key == f'Face_Recog_Key_Block_{block_num}_Run_{run_num}':
            recog_response_data_face.append([image_start_time, image_time, 1.000])
        elif recog_key == f'Place_Recog_Key_Block_{block_num}_Run_{run_num}':
            recog_response_data_place.append([image_start_time, image_time, 1.000])
        elif recog_key == f'Pair_Recog_Key_Block_{block_num}_Run_{run_num}':
            recog_response_data_pair.append([image_start_time, image_time, 1.000])

    else: 
        # If the response is incorrect or no response, remove the data for the image from encoding txt response
        
        # Modify the recognition image path to match the encoding phase image path
        encoding_image_path = current_displayed_image.replace('_Recog', '_Encoding')

        if recog_key == f'Face_Recog_Key_Block_{block_num}_Run_{run_num}':
            if encoding_image_path in encoding_response_data_face:
                encoding_response_data_face.pop(encoding_image_path, None)
        elif recog_key == f'Place_Recog_Key_Block_{block_num}_Run_{run_num}':
            if encoding_image_path in encoding_response_data_place:
                encoding_response_data_place.pop(encoding_image_path, None)
        elif recog_key == f'Pair_Recog_Key_Block_{block_num}_Run_{run_num}':
            if encoding_image_path in encoding_response_data_pair:
                encoding_response_data_pair.pop(encoding_image_path, None)


# Create a keyboard object and a timer for the keyboard
kb = keyboard.Keyboard()
timer_key = core.Clock()

start_time_key = 0
current_time_key = 0

# Initialize the recognition lists to store recognition data
Recog_Data_Face_Block_1_Run_1 = []
Recog_Data_Place_Block_1_Run_1 = []
Recog_Data_Pair_Block_1_Run_1 = []

# Initialize the encoding lists to store encoding txt data based on recognition responses
Recog_Response_Data_Face_Block_1_Run_1 = []
Recog_Response_Data_Place_Block_1_Run_1 = []
Recog_Response_Data_Pair_Block_1_Run_1 = []
recogTrialRun1_Image = visual.ImageStim(
    win=win,
    name='recogTrialRun1_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
recogTrialTestKeyRespRun1 = keyboard.Keyboard()

# --- Initialize components for Routine "triggerSync" ---
triggerSyncText = visual.TextStim(win=win, name='triggerSyncText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
triggerSync_Key_Resp = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "encodeTrial_Run2" ---
# Run 'Begin Experiment' code from encodeTrialRun2Code
from psychopy.hardware import keyboard
from psychopy import core

# Function to update encoding txt lists 
def update_encoding_data(
    block_num,
    run_num,
    random_encode_key,
    encode_image_started_time,
    image_time,
    current_displayed_image,
    encoding_data_face,
    encoding_data_place,
    encoding_data_pair,
    encoding_response_data_face,
    encoding_response_data_place,
    encoding_response_data_pair,
):

    # The encoding data to add
    encoding_data_to_add = [encode_image_started_time, image_time, 1.000]
    
    if random_encode_key == f'Face_Encode_Key_Block_{block_num}_Run_{run_num}':
        encoding_data_face.append(encoding_data_to_add)
        encoding_response_data_face[current_displayed_image] = encoding_data_to_add
    elif random_encode_key == f'Place_Encode_Key_Block_{block_num}_Run_{run_num}':
        encoding_data_place.append(encoding_data_to_add)
        encoding_response_data_place[current_displayed_image] = encoding_data_to_add
    elif random_encode_key == f'Pair_Encode_Key_Block_{block_num}_Run_{run_num}':
        encoding_data_pair.append(encoding_data_to_add)
        encoding_response_data_pair[current_displayed_image] = encoding_data_to_add

kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Initialize the encoding lists to store encoding txt data
Encoding_Data_Face_Block_2_Run_2 = []
Encoding_Data_Place_Block_2_Run_2 = []
Encoding_Data_Pair_Block_2_Run_2 = []

# Initialize the encoding response data as dictionaries
Encoding_Response_Data_Face_Block_2_Run_2 = {}
Encoding_Response_Data_Place_Block_2_Run_2 = {}
Encoding_Response_Data_Pair_Block_2_Run_2 = {}

# Initialize the image_start_time variable
image_start_time = 0

encodeTrialRun2_Image = visual.ImageStim(
    win=win,
    name='encodeTrialRun2_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
encodeTrialTestKeyRespRun2 = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "recognitionTrial_Run2" ---
# Run 'Begin Experiment' code from recognitonTrialRun2Code
from psychopy.hardware import keyboard
from psychopy import core

# Create a keyboard object and a timer for the keyboard
kb = keyboard.Keyboard()
timer_key = core.Clock()

start_time_key = 0
current_time_key = 0

# Initialize the recognition lists to store recognition data
Recog_Data_Face_Block_2_Run_2 = []
Recog_Data_Place_Block_2_Run_2 = []
Recog_Data_Pair_Block_2_Run_2 = []

# Initialize the encoding lists to store encoding txt data based on recognition responses
Recog_Response_Data_Face_Block_2_Run_2 = []
Recog_Response_Data_Place_Block_2_Run_2 = []
Recog_Response_Data_Pair_Block_2_Run_2 = []

recogTrialRun2_Image = visual.ImageStim(
    win=win,
    name='recogTrialRun2_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
recogTrialTestKeyRespRun2 = keyboard.Keyboard()

# --- Initialize components for Routine "triggerSync" ---
triggerSyncText = visual.TextStim(win=win, name='triggerSyncText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
triggerSync_Key_Resp = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "encodeTrial_Run3" ---
# Run 'Begin Experiment' code from encodeTrialRun3Code
from psychopy.hardware import keyboard
from psychopy import core

kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Initialize the encoding lists to store encoding txt data
Encoding_Data_Face_Block_3_Run_3 = []
Encoding_Data_Place_Block_3_Run_3 = []
Encoding_Data_Pair_Block_3_Run_3 = []

# Initialize the encoding response data as dictionaries
Encoding_Response_Data_Face_Block_3_Run_3 = {}
Encoding_Response_Data_Place_Block_3_Run_3 = {}
Encoding_Response_Data_Pair_Block_3_Run_3 = {}

# Initialize the image_start_time variable
image_start_time = 0


encodeTrialRun3_Image = visual.ImageStim(
    win=win,
    name='encodeTrialRun3_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
encodeTrialTestKeyRespRun3 = keyboard.Keyboard()

# --- Initialize components for Routine "fixationCross" ---
# Run 'Begin Experiment' code from fixationCode
from psychopy.hardware import keyboard
from psychopy import core

# Function for ISI duration and display
def set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                             Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                             Encoding_Started, Recognition_Started,
                             Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                             Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                             Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3):

    # Initialize variables
    ISI_Message = " "
    ISI_Time = 0

    # Set ISI Time and Message
    if Fix_Cross_Started == True:
        ISI_Message = "+"
        ISI_Time = jitter_duration[current_jitter_index]
    else:
        ISI_Time = 10.0  # Instruction screen time

        # Block logic
        blocks_started = [Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started]
        random_encode_keys = [Random_Encode_Key_Block_1_Run_1, Random_Encode_Key_Block_2_Run_2, Random_Encode_Key_Block_3_Run_3]
        recog_keys = [Recog_Key_Block_1_Run_1, Recog_Key_Block_2_Run_2, Recog_Key_Block_3_Run_3]

        for i in range(len(blocks_started)):
            if blocks_started[i] and not any(blocks_started[:i] + blocks_started[i+1:]):
                if Encoding_Started:
                    if random_encode_keys[i] == f'Face_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Encoding_Text
                    elif random_encode_keys[i] == f'Place_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Encoding_Text
                    elif random_encode_keys[i] == f'Pair_Encode_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Encoding_Text

                elif Recognition_Started:
                    if recog_keys[i] == f'Face_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Face_Recog_Text
                    elif recog_keys[i] == f'Place_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Place_Recog_Text
                    elif recog_keys[i] == f'Pair_Recog_Key_Block_{i+1}_Run_{i+1}':
                        ISI_Message = Pair_Recog_Text

    return ISI_Time, ISI_Message


kb = keyboard.Keyboard()
timer_key = core.Clock()
start_time_key = 0
current_time_key = 0

# Assumes instruction screen is first which is 10.0 s
ISI_Time = 10.0 if Fix_Cross_Started == False else jitter_duration[current_jitter_index]

# ISI text for instructions 
# Face encoding condition text
Face_Encoding_Text = """
Male or female?

"""
# Place encoding condition text
Place_Encoding_Text = """
Is there water in the scene?

"""

# Pair encoding condition text
Pair_Encoding_Text = """
Does the face "fit" with the scene?

"""

# ISI text for instruction screen 
# Face recognition condition text
Face_Recog_Text = """
Was this face shown before?

"""
# Place recognition condition text
Place_Recog_Text = """
Was this scene shown before?

"""

# Pair recognition condition text
Pair_Recog_Text = """
Was this face shown together with this scene before?

"""
fixationText = visual.TextStim(win=win, name='fixationText',
    text='',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "recognitionTrial_Run3" ---
# Run 'Begin Experiment' code from recognitonTrialRun3Code
from psychopy.hardware import keyboard
from psychopy import core

# Create a keyboard object and a timer for the keyboard
kb = keyboard.Keyboard()
timer_key = core.Clock()

start_time_key = 0
current_time_key = 0

# Initialize the recognition lists to store recognition data
Recog_Data_Face_Block_3_Run_3 = []
Recog_Data_Place_Block_3_Run_3 = []
Recog_Data_Pair_Block_3_Run_3 = []

# Initialize the encoding lists to store encoding txt data based on recognition responses
Recog_Response_Data_Face_Block_3_Run_3 = []
Recog_Response_Data_Place_Block_3_Run_3 = []
Recog_Response_Data_Pair_Block_3_Run_3 = []


recogTrialRun3_Image = visual.ImageStim(
    win=win,
    name='recogTrialRun3_Image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.33, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
recogTrialTestKeyRespRun3 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "setUp" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
setUpComponents = []
for thisComponent in setUpComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "setUp" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in setUpComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "setUp" ---
for thisComponent in setUpComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "setUp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
condLoaderLoop = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('conditions.xlsx'),
    seed=None, name='condLoaderLoop')
thisExp.addLoop(condLoaderLoop)  # add the loop to the experiment
thisCondLoaderLoop = condLoaderLoop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisCondLoaderLoop.rgb)
if thisCondLoaderLoop != None:
    for paramName in thisCondLoaderLoop:
        exec('{} = thisCondLoaderLoop[paramName]'.format(paramName))

for thisCondLoaderLoop in condLoaderLoop:
    currentLoop = condLoaderLoop
    # abbreviate parameter names if possible (e.g. rgb = thisCondLoaderLoop.rgb)
    if thisCondLoaderLoop != None:
        for paramName in thisCondLoaderLoop:
            exec('{} = thisCondLoaderLoop[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "condLoader" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from condLoaderCod
    import random
    
    # BLOCK 1 RUN 1
    # Create a list of encoding dictionary keys and shuffle them
    Encode_Key_List_Block_1_Run_1 = list(Encoding_Dictionary_Block_1_Run_1.keys())
    random.shuffle(Encode_Key_List_Block_1_Run_1 )
    
    # Get a random encoding key and its corresponding encoding list from the shuffled keys
    Random_Encode_Key_Block_1_Run_1 = Encode_Key_List_Block_1_Run_1[current_encode_key_index]
    Random_Encode_List_Block_1_Run_1 = Encoding_Dictionary_Block_1_Run_1[Random_Encode_Key_Block_1_Run_1]
    
    # Determine the corresponding recognition key based on the chosen encoding key
    if Random_Encode_Key_Block_1_Run_1  == 'Face_Encode_Key_Block_1_Run_1':
        Recog_Key_Block_1_Run_1 = 'Face_Recog_Key_Block_1_Run_1'
    elif Random_Encode_Key_Block_1_Run_1 == 'Place_Encode_Key_Block_1_Run_1':
        Recog_Key_Block_1_Run_1 = 'Place_Recog_Key_Block_1_Run_1'
    elif Random_Encode_Key_Block_1_Run_1 == 'Pair_Encode_Key_Block_1_Run_1':
        Recog_Key_Block_1_Run_1 = 'Pair_Recog_Key_Block_1_Run_1'
    
    # Get the corresponding recognition list based on the determined recognition key
    Recog_List_Block_1_Run_1 = Recognition_Dictionary_Block_1_Run_1[Recog_Key_Block_1_Run_1]
    
    # BLOCK 2 RUN 2
    # Create a list of encoding dictionary keys and shuffle them
    Encode_Key_List_Block_2_Run_2 = list(Encoding_Dictionary_Block_2_Run_2.keys())
    random.shuffle(Encode_Key_List_Block_2_Run_2)
    
    # Get a random encoding key and its corresponding encoding list from the shuffled keys
    Random_Encode_Key_Block_2_Run_2 = Encode_Key_List_Block_2_Run_2[current_encode_key_index]
    Random_Encode_List_Block_2_Run_2 = Encoding_Dictionary_Block_2_Run_2[Random_Encode_Key_Block_2_Run_2]
    
    # Determine the corresponding recognition key based on the chosen encoding key
    if Random_Encode_Key_Block_2_Run_2  == 'Face_Encode_Key_Block_2_Run_2':
        Recog_Key_Block_2_Run_2 = 'Face_Recog_Key_Block_2_Run_2'
    elif Random_Encode_Key_Block_2_Run_2 == 'Place_Encode_Key_Block_2_Run_2':
        Recog_Key_Block_2_Run_2 = 'Place_Recog_Key_Block_2_Run_2'
    elif Random_Encode_Key_Block_2_Run_2 == 'Pair_Encode_Key_Block_2_Run_2':
        Recog_Key_Block_2_Run_2 = 'Pair_Recog_Key_Block_2_Run_2'
    
    # Get the corresponding recognition list based on the determined recognition key
    Recog_List_Block_2_Run_2 = Recognition_Dictionary_Block_2_Run_2[Recog_Key_Block_2_Run_2]
    
    # BLOCK 3 RUN 3
    # Create a list of encoding dictionary keys and shuffle them
    Encode_Key_List_Block_3_Run_3 = list(Encoding_Dictionary_Block_3_Run_3.keys())
    random.shuffle(Encode_Key_List_Block_3_Run_3)
    
    # Get a random encoding key and its corresponding encoding list from the shuffled keys
    Random_Encode_Key_Block_3_Run_3 = Encode_Key_List_Block_3_Run_3[current_encode_key_index]
    Random_Encode_List_Block_3_Run_3 = Encoding_Dictionary_Block_3_Run_3[Random_Encode_Key_Block_3_Run_3]
    
    # Determine the corresponding recognition key based on the chosen encoding key
    if Random_Encode_Key_Block_3_Run_3  == 'Face_Encode_Key_Block_3_Run_3':
        Recog_Key_Block_3_Run_3 = 'Face_Recog_Key_Block_3_Run_3'
    elif Random_Encode_Key_Block_3_Run_3 == 'Place_Encode_Key_Block_3_Run_3':
        Recog_Key_Block_3_Run_3 = 'Place_Recog_Key_Block_3_Run_3'
    elif Random_Encode_Key_Block_3_Run_3 == 'Pair_Encode_Key_Block_3_Run_3':
        Recog_Key_Block_3_Run_3 = 'Pair_Recog_Key_Block_3_Run_3'
    
    # Get the corresponding recognition list based on the determined recognition key
    Recog_List_Block_3_Run_3 = Recognition_Dictionary_Block_3_Run_3[Recog_Key_Block_3_Run_3]
    
    # keep track of which components have finished
    condLoaderComponents = []
    for thisComponent in condLoaderComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "condLoader" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in condLoaderComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "condLoader" ---
    for thisComponent in condLoaderComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "condLoader" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1.0 repeats of 'condLoaderLoop'


# --- Prepare to start Routine "triggerSync" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# Run 'Begin Routine' code from triggerSyncCode
trigger_sync_message = "+"
if volume_counter == 1:
    trigger_sync_message = "Waiting for scanner..."




triggerSyncText.setText(trigger_sync_message)
triggerSync_Key_Resp.keys = []
triggerSync_Key_Resp.rt = []
_triggerSync_Key_Resp_allKeys = []
# keep track of which components have finished
triggerSyncComponents = [triggerSyncText, triggerSync_Key_Resp]
for thisComponent in triggerSyncComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "triggerSync" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *triggerSyncText* updates
    if triggerSyncText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSyncText.frameNStart = frameN  # exact frame index
        triggerSyncText.tStart = t  # local t and not account for scr refresh
        triggerSyncText.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSyncText, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'triggerSyncText.started')
        triggerSyncText.setAutoDraw(True)
    
    # *triggerSync_Key_Resp* updates
    waitOnFlip = False
    if triggerSync_Key_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSync_Key_Resp.frameNStart = frameN  # exact frame index
        triggerSync_Key_Resp.tStart = t  # local t and not account for scr refresh
        triggerSync_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSync_Key_Resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'triggerSync_Key_Resp.started')
        triggerSync_Key_Resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(triggerSync_Key_Resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(triggerSync_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if triggerSync_Key_Resp.status == STARTED and not waitOnFlip:
        theseKeys = triggerSync_Key_Resp.getKeys(keyList=['5','t','s'], waitRelease=False)
        _triggerSync_Key_Resp_allKeys.extend(theseKeys)
        if len(_triggerSync_Key_Resp_allKeys):
            triggerSync_Key_Resp.keys = _triggerSync_Key_Resp_allKeys[-1].name  # just the last key pressed
            triggerSync_Key_Resp.rt = _triggerSync_Key_Resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in triggerSyncComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "triggerSync" ---
for thisComponent in triggerSyncComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if triggerSync_Key_Resp.keys in ['', [], None]:  # No response was made
    triggerSync_Key_Resp.keys = None
thisExp.addData('triggerSync_Key_Resp.keys',triggerSync_Key_Resp.keys)
if triggerSync_Key_Resp.keys != None:  # we had a response
    thisExp.addData('triggerSync_Key_Resp.rt', triggerSync_Key_Resp.rt)
thisExp.nextEntry()
# the Routine "triggerSync" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Block1Loop = data.TrialHandler(nReps=3.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Block1Loop')
thisExp.addLoop(Block1Loop)  # add the loop to the experiment
thisBlock1Loop = Block1Loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock1Loop.rgb)
if thisBlock1Loop != None:
    for paramName in thisBlock1Loop:
        exec('{} = thisBlock1Loop[paramName]'.format(paramName))

for thisBlock1Loop in Block1Loop:
    currentLoop = Block1Loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlock1Loop.rgb)
    if thisBlock1Loop != None:
        for paramName in thisBlock1Loop:
            exec('{} = thisBlock1Loop[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    encodeLoop_Block1Run1 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='encodeLoop_Block1Run1')
    thisExp.addLoop(encodeLoop_Block1Run1)  # add the loop to the experiment
    thisEncodeLoop_Block1Run1 = encodeLoop_Block1Run1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block1Run1.rgb)
    if thisEncodeLoop_Block1Run1 != None:
        for paramName in thisEncodeLoop_Block1Run1:
            exec('{} = thisEncodeLoop_Block1Run1[paramName]'.format(paramName))
    
    for thisEncodeLoop_Block1Run1 in encodeLoop_Block1Run1:
        currentLoop = encodeLoop_Block1Run1
        # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block1Run1.rgb)
        if thisEncodeLoop_Block1Run1 != None:
            for paramName in thisEncodeLoop_Block1Run1:
                exec('{} = thisEncodeLoop_Block1Run1[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time) # Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 1 RUN 1 ENCODING
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationText.started')
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 1 RUN 1 ENCODING
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationText.stopped')
                    fixationText.setAutoDraw(False)
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 1 RUN 1 ENCODING
        
        # --- Prepare to start Routine "encodeTrial_Run1" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from encodeTrialRun1Code
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Random_Encode_List_Block_1_Run_1[current_encode_index]
        
        # Add the image_file data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_1_Run)
        
        
        encodeTrialTestKeyRespRun1.keys = []
        encodeTrialTestKeyRespRun1.rt = []
        _encodeTrialTestKeyRespRun1_allKeys = []
        # keep track of which components have finished
        encodeTrial_Run1Components = [encodeTrialRun1_Image, encodeTrialTestKeyRespRun1]
        for thisComponent in encodeTrial_Run1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encodeTrial_Run1" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from encodeTrialRun1Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            # Check if the image should start and add to encoding lists for txt
            if encodeTrialRun1_Image.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                image_start_time = tThisFlipGlobal
                if Random_Encode_Key_Block_1_Run_1  == 'Face_Encode_Key_Block_1_Run_1':
                    Encoding_Data_Face_Block_1_Run_1.append([image_start_time, image_time, 1.000])
                    Encoding_Response_Data_Face_Block_1_Run_1[current_displayed_image] = [image_start_time, image_time, 1.000]
                elif Random_Encode_Key_Block_1_Run_1 == 'Place_Encode_Key_Block_1_Run_1':
                    Encoding_Data_Place_Block_1_Run_1.append([image_start_time, image_time, 1.000])
                    Encoding_Response_Data_Place_Block_1_Run_1[current_displayed_image] = [image_start_time, image_time, 1.000]
                elif Random_Encode_Key_Block_1_Run_1 == 'Pair_Encode_Key_Block_1_Run_1':
                    Encoding_Data_Pair_Block_1_Run_1.append([image_start_time, image_time, 1.000])
                    Encoding_Response_Data_Pair_Block_1_Run_1[current_displayed_image] = [image_start_time, image_time, 1.000]
            
            
            
            
            
            
            # *encodeTrialRun1_Image* updates
            if encodeTrialRun1_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialRun1_Image.frameNStart = frameN  # exact frame index
                encodeTrialRun1_Image.tStart = t  # local t and not account for scr refresh
                encodeTrialRun1_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialRun1_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'encodeTrialRun1_Image.started')
                encodeTrialRun1_Image.setAutoDraw(True)
            if encodeTrialRun1_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialRun1_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialRun1_Image.tStop = t  # not accounting for scr refresh
                    encodeTrialRun1_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encodeTrialRun1_Image.stopped')
                    encodeTrialRun1_Image.setAutoDraw(False)
            if encodeTrialRun1_Image.status == STARTED:  # only update if drawing
                encodeTrialRun1_Image.setImage(Random_Encode_List_Block_1_Run_1[current_encode_index], log=False)
            
            # *encodeTrialTestKeyRespRun1* updates
            waitOnFlip = False
            if encodeTrialTestKeyRespRun1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialTestKeyRespRun1.frameNStart = frameN  # exact frame index
                encodeTrialTestKeyRespRun1.tStart = t  # local t and not account for scr refresh
                encodeTrialTestKeyRespRun1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialTestKeyRespRun1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun1.started')
                encodeTrialTestKeyRespRun1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(encodeTrialTestKeyRespRun1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(encodeTrialTestKeyRespRun1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if encodeTrialTestKeyRespRun1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialTestKeyRespRun1.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialTestKeyRespRun1.tStop = t  # not accounting for scr refresh
                    encodeTrialTestKeyRespRun1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun1.stopped')
                    encodeTrialTestKeyRespRun1.status = FINISHED
            if encodeTrialTestKeyRespRun1.status == STARTED and not waitOnFlip:
                theseKeys = encodeTrialTestKeyRespRun1.getKeys(keyList=['1','2'], waitRelease=False)
                _encodeTrialTestKeyRespRun1_allKeys.extend(theseKeys)
                if len(_encodeTrialTestKeyRespRun1_allKeys):
                    encodeTrialTestKeyRespRun1.keys = _encodeTrialTestKeyRespRun1_allKeys[-1].name  # just the last key pressed
                    encodeTrialTestKeyRespRun1.rt = _encodeTrialTestKeyRespRun1_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encodeTrial_Run1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encodeTrial_Run1" ---
        for thisComponent in encodeTrial_Run1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from encodeTrialRun1Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        # Record encoding response
        gender_is_correct, liquid_is_correct, pair_fit_is_correct, image_gender, response_gender, image_water, response_water, response_pair_fit = update_encode_pre_recog_response_data(
            block_num=1,
            run_num=Block_1_Run,
            current_displayed_image=current_displayed_image,
            random_encode_key=Random_Encode_Key_Block_1_Run_1,
            response_key=encodeTrialTestKeyRespRun1.keys,
            liquid_conditions=["beach12", "city5", "citybeach2", "citybeach3", "stream2"]
        )
        
        # Save encoding responses to output
        if gender_is_correct is not None:
            thisExp.addData('image_gender', image_gender)
            thisExp.addData('response_gender', response_gender)
            thisExp.addData('gender_is_correct', gender_is_correct)
        
        if liquid_is_correct is not None:
            thisExp.addData('image_water', image_water)
            thisExp.addData('response_water', response_water)
            thisExp.addData('liquid_is_correct', liquid_is_correct)
        
        if response_pair_fit is not None:
            thisExp.addData('response_pair_fit', response_pair_fit)
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block1Loop.thisRepN)
        
        # Increment the current encode index to iterate over images
        current_encode_index += 1
        
        if current_encode_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            encode_trials_counter += 1
        
        if current_encode_index == 12: # Last stimulus is shown
            
            current_encode_index = 0 # Reset stimulus counter
            current_encode_key_index += 1 # Next encode condition
            current_jitter_index = 0 # Reset current jitter index
            
            Encoding_Started = False # End of encoding 
            Fix_Cross_Started = False # Next ISI is instructions
            Recognition_Started = True # Start of recognition
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            if current_encode_key_index == 3: # At last encoding condition
                current_encode_key_index = 0 # Reset encode condition order
            else:
                Random_Encode_Key_Block_1_Run_1 = Encode_Key_List_Block_1_Run_1[current_encode_key_index]
                Random_Encode_List_Block_1_Run_1 = Encoding_Dictionary_Block_1_Run_1[Random_Encode_Key_Block_1_Run_1]
         
        if Fix_Cross_Started == True and current_encode_index > 1 : # Iterate through jitter times if showing fixation cross and after first image
            current_jitter_index += 1
         
        
        # check responses
        if encodeTrialTestKeyRespRun1.keys in ['', [], None]:  # No response was made
            encodeTrialTestKeyRespRun1.keys = None
        encodeLoop_Block1Run1.addData('encodeTrialTestKeyRespRun1.keys',encodeTrialTestKeyRespRun1.keys)
        if encodeTrialTestKeyRespRun1.keys != None:  # we had a response
            encodeLoop_Block1Run1.addData('encodeTrialTestKeyRespRun1.rt', encodeTrialTestKeyRespRun1.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'encodeLoop_Block1Run1'
    
    
    # set up handler to look after randomisation of conditions etc
    recognitionLoop_Block1Run1 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='recognitionLoop_Block1Run1')
    thisExp.addLoop(recognitionLoop_Block1Run1)  # add the loop to the experiment
    thisRecognitionLoop_Block1Run1 = recognitionLoop_Block1Run1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block1Run1.rgb)
    if thisRecognitionLoop_Block1Run1 != None:
        for paramName in thisRecognitionLoop_Block1Run1:
            exec('{} = thisRecognitionLoop_Block1Run1[paramName]'.format(paramName))
    
    for thisRecognitionLoop_Block1Run1 in recognitionLoop_Block1Run1:
        currentLoop = recognitionLoop_Block1Run1
        # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block1Run1.rgb)
        if thisRecognitionLoop_Block1Run1 != None:
            for paramName in thisRecognitionLoop_Block1Run1:
                exec('{} = thisRecognitionLoop_Block1Run1[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time)# Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 1 RUN 1 RECOGNITION
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationText.started')
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 1 RUN 1 RECOGNITION
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationText.stopped')
                    fixationText.setAutoDraw(False)
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 1 RUN 1 RECOGNITION
        
        # --- Prepare to start Routine "recognitionTrial_Run1" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from recognitonTrialRun1Code
        # Add the start time and volume counter to the data
        thisExp.addData('start_volume', volume_counter)
        
        # Reset the keyboard clock
        kb.clock.reset()  # when you want to start the timer from
        
        # Set the volume counter message
        volume_counter_message = str(volume_counter) + " out of " + str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Recog_List_Block_1_Run_1[current_recog_index]
        
        # Add the current_displayed_image to the data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_1_Run)
        
        
        recogTrialTestKeyRespRun1.keys = []
        recogTrialTestKeyRespRun1.rt = []
        _recogTrialTestKeyRespRun1_allKeys = []
        # keep track of which components have finished
        recognitionTrial_Run1Components = [recogTrialRun1_Image, recogTrialTestKeyRespRun1]
        for thisComponent in recognitionTrial_Run1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recognitionTrial_Run1" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from recognitonTrialRun1Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            if recogTrialRun1_Image.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                image_start_time = tThisFlipGlobal
                if Recog_Key_Block_1_Run_1 == 'Face_Recog_Key_Block_1_Run_1':
                    Recog_Data_Face_Block_1_Run_1.append([image_start_time, image_time, 1.000])
                elif Recog_Key_Block_1_Run_1 == 'Place_Recog_Key_Block_1_Run_1':
                    Recog_Data_Place_Block_1_Run_1.append([image_start_time, image_time, 1.000])
                elif Recog_Key_Block_1_Run_1 == 'Pair_Recog_Key_Block_1_Run_1':
                    Recog_Data_Pair_Block_1_Run_1.append([image_start_time, image_time, 1.000])
            
            # *recogTrialRun1_Image* updates
            if recogTrialRun1_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialRun1_Image.frameNStart = frameN  # exact frame index
                recogTrialRun1_Image.tStart = t  # local t and not account for scr refresh
                recogTrialRun1_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialRun1_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recogTrialRun1_Image.started')
                recogTrialRun1_Image.setAutoDraw(True)
            if recogTrialRun1_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialRun1_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialRun1_Image.tStop = t  # not accounting for scr refresh
                    recogTrialRun1_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recogTrialRun1_Image.stopped')
                    recogTrialRun1_Image.setAutoDraw(False)
            if recogTrialRun1_Image.status == STARTED:  # only update if drawing
                recogTrialRun1_Image.setImage(Recog_List_Block_1_Run_1[current_recog_index], log=False)
            
            # *recogTrialTestKeyRespRun1* updates
            waitOnFlip = False
            if recogTrialTestKeyRespRun1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialTestKeyRespRun1.frameNStart = frameN  # exact frame index
                recogTrialTestKeyRespRun1.tStart = t  # local t and not account for scr refresh
                recogTrialTestKeyRespRun1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialTestKeyRespRun1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun1.started')
                recogTrialTestKeyRespRun1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recogTrialTestKeyRespRun1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recogTrialTestKeyRespRun1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recogTrialTestKeyRespRun1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialTestKeyRespRun1.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialTestKeyRespRun1.tStop = t  # not accounting for scr refresh
                    recogTrialTestKeyRespRun1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun1.stopped')
                    recogTrialTestKeyRespRun1.status = FINISHED
            if recogTrialTestKeyRespRun1.status == STARTED and not waitOnFlip:
                theseKeys = recogTrialTestKeyRespRun1.getKeys(keyList=['1','2'], waitRelease=False)
                _recogTrialTestKeyRespRun1_allKeys.extend(theseKeys)
                if len(_recogTrialTestKeyRespRun1_allKeys):
                    recogTrialTestKeyRespRun1.keys = _recogTrialTestKeyRespRun1_allKeys[-1].name  # just the last key pressed
                    recogTrialTestKeyRespRun1.rt = _recogTrialTestKeyRespRun1_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recognitionTrial_Run1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recognitionTrial_Run1" ---
        for thisComponent in recognitionTrial_Run1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from recognitonTrialRun1Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        # Get the image OLD/NEW status and determine if the response is correct
        image_name = Recog_List_Block_1_Run_1[current_recog_index]
        if 'OLD' in image_name:
            image_old_new = "OLD"
        else:
            image_old_new = "NEW"
        # Add the image_old_new data to the experiment handler before response_old_new
        thisExp.addData('image_old_new', image_old_new)
        
        # Get the participant's response for the recognition trial
        if recogTrialTestKeyRespRun1.keys == '1':
            response_old_new = "OLD"
        elif recogTrialTestKeyRespRun1.keys == '2':
            response_old_new = "NEW"
        else:
            response_old_new = "NONE"
        # Add the response_old_new data to the experiment handler
        thisExp.addData('response_old_new', response_old_new)
        
        # Determine d prime and add results to the experiment handler
        is_correct = d_prime(image_old_new=image_old_new, response_key=recogTrialTestKeyRespRun1.keys)
        thisExp.addData('is_correct', is_correct)
        
        # Update encoding and recognition txt response list
        update_encoding_recognition_response_data(
            block_num=1, # Change based on block number
            run_num=Block_1_Run,  # Change based on run number
            is_correct=is_correct, 
            current_displayed_image=current_displayed_image, 
            recog_key=Recog_Key_Block_1_Run_1, # Change based on Block and condition
            image_start_time=image_start_time, 
            image_time=image_time, 
            recog_response_data_face=Recog_Response_Data_Face_Block_1_Run_1,  # Change based on Block and condition
            recog_response_data_place=Recog_Response_Data_Place_Block_1_Run_1,  # Change based on Block and condition
            recog_response_data_pair=Recog_Response_Data_Pair_Block_1_Run_1,  # Change based on Block and condition
            encoding_response_data_face=Encoding_Response_Data_Face_Block_1_Run_1,  # Change based on Block and condition
            encoding_response_data_place=Encoding_Response_Data_Place_Block_1_Run_1,  # Change based on Block and condition
            encoding_response_data_pair=Encoding_Response_Data_Pair_Block_1_Run_1  # Change based on Block and condition
        )
        
        # Increment the current recognition index to iterate over images
        current_recog_index += 1
        
        if current_recog_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            recognition_trials_counter += 1
            
        if current_recog_index == 12: # Last stimulus is shown
            # Increment the number of trialis in block loop
            block_loop_trials_counter += 1
            
            Recognition_Started = False # End of recognition
            Fix_Cross_Started = False # Next ISI is instructions
            Encoding_Started = True # Start of encoding 
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            # Reset the current recognition index 
            current_recog_index = 0
            
            # Reset current jitter index
            current_jitter_index = 0
            
            # Determine the corresponding recognition key based on the chosen encoding key
            if Random_Encode_Key_Block_1_Run_1  == 'Face_Encode_Key_Block_1_Run_1':
                Recog_Key_Block_1_Run_1 = 'Face_Recog_Key_Block_1_Run_1'
            elif Random_Encode_Key_Block_1_Run_1 == 'Place_Encode_Key_Block_1_Run_1':
                Recog_Key_Block_1_Run_1 = 'Place_Recog_Key_Block_1_Run_1'
            elif Random_Encode_Key_Block_1_Run_1 == 'Pair_Encode_Key_Block_1_Run_1':
                Recog_Key_Block_1_Run_1 = 'Pair_Recog_Key_Block_1_Run_1'
        
            # Get the corresponding recognition list based on the determined recognition key
            Recog_List_Block_1_Run_1 = Recognition_Dictionary_Block_1_Run_1[Recog_Key_Block_1_Run_1]
        
        # Iterate through jitter times if showing fixation cross and after first image
        if Fix_Cross_Started == True and current_recog_index > 1 : 
            current_jitter_index += 1
         
        # Reset the trial counters
        if block_loop_trials_counter == 3:
            block_loop_trials_counter = 0
            recognition_trials_counter = 0
            encode_trials_counter = 0
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block1Loop.thisRepN)
        
        # When Block Loop ends based on stimuli total
        if len(Block_Loop_Voulme_Counter) == 72: 
            volume_counter = 1 # Reset volume_counter
            Block_Loop_Voulme_Counter = [] # Reinitalize counter
            Block_1_Run_1_Started = False # Block 1 Run 1 finished
            Encoding_Started = True # Start of encoding for next block 
            Recognition_Started = False # End of recognition for next block
            Block_2_Run_2_Started = True # Begin next block
        
        # check responses
        if recogTrialTestKeyRespRun1.keys in ['', [], None]:  # No response was made
            recogTrialTestKeyRespRun1.keys = None
        recognitionLoop_Block1Run1.addData('recogTrialTestKeyRespRun1.keys',recogTrialTestKeyRespRun1.keys)
        if recogTrialTestKeyRespRun1.keys != None:  # we had a response
            recognitionLoop_Block1Run1.addData('recogTrialTestKeyRespRun1.rt', recogTrialTestKeyRespRun1.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'recognitionLoop_Block1Run1'
    
    thisExp.nextEntry()
    
# completed 3.0 repeats of 'Block1Loop'


# --- Prepare to start Routine "triggerSync" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# Run 'Begin Routine' code from triggerSyncCode
trigger_sync_message = "+"
if volume_counter == 1:
    trigger_sync_message = "Waiting for scanner..."




triggerSyncText.setText(trigger_sync_message)
triggerSync_Key_Resp.keys = []
triggerSync_Key_Resp.rt = []
_triggerSync_Key_Resp_allKeys = []
# keep track of which components have finished
triggerSyncComponents = [triggerSyncText, triggerSync_Key_Resp]
for thisComponent in triggerSyncComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1
Time_Since_Run.reset()  # Reset clock for next run onset and stop times
next_fixCross_started_time = 0  # Initalize fixation cross start time, onset times depend on this

# --- Run Routine "triggerSync" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *triggerSyncText* updates
    if triggerSyncText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSyncText.frameNStart = frameN  # exact frame index
        triggerSyncText.tStart = t  # local t and not account for scr refresh
        triggerSyncText.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSyncText, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
#        thisExp.timestampOnFlip(win, 'triggerSyncText.started') # Original code which tracks internal clock (which can't be reset)
        triggerSync_started_time = Time_Since_Run.getTime()
        thisExp.addData('triggerSyncText.started', triggerSync_started_time) # triggerSyncText.started for Block 2 Run 2
        triggerSyncText.setAutoDraw(True)
    
    # *triggerSync_Key_Resp* updates
    waitOnFlip = False
    if triggerSync_Key_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSync_Key_Resp.frameNStart = frameN  # exact frame index
        triggerSync_Key_Resp.tStart = t  # local t and not account for scr refresh
        triggerSync_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSync_Key_Resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        #        thisExp.timestampOnFlip(win, 'triggerSync_Key_Resp.started') # Original code which tracks internal clock (which can't be reset)
        thisExp.addData('triggerSync_Key_Resp.started', triggerSync_started_time) # triggerSync_Key_Resp.started for Block 2 Run 2
        triggerSync_Key_Resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(triggerSync_Key_Resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(triggerSync_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if triggerSync_Key_Resp.status == STARTED and not waitOnFlip:
        theseKeys = triggerSync_Key_Resp.getKeys(keyList=['5','t','s'], waitRelease=False)
        _triggerSync_Key_Resp_allKeys.extend(theseKeys)
        if len(_triggerSync_Key_Resp_allKeys):
            triggerSync_Key_Resp.keys = _triggerSync_Key_Resp_allKeys[-1].name  # just the last key pressed
            triggerSync_Key_Resp.rt = _triggerSync_Key_Resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in triggerSyncComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "triggerSync" ---
for thisComponent in triggerSyncComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if triggerSync_Key_Resp.keys in ['', [], None]:  # No response was made
    triggerSync_Key_Resp.keys = None
thisExp.addData('triggerSync_Key_Resp.keys',triggerSync_Key_Resp.keys)
if triggerSync_Key_Resp.keys != None:  # we had a response
    thisExp.addData('triggerSync_Key_Resp.rt', triggerSync_Key_Resp.rt)
thisExp.nextEntry()
# the Routine "triggerSync" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Block2Loop = data.TrialHandler(nReps=3.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Block2Loop')
thisExp.addLoop(Block2Loop)  # add the loop to the experiment
thisBlock2Loop = Block2Loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock2Loop.rgb)
if thisBlock2Loop != None:
    for paramName in thisBlock2Loop:
        exec('{} = thisBlock2Loop[paramName]'.format(paramName))

for thisBlock2Loop in Block2Loop:
    currentLoop = Block2Loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlock2Loop.rgb)
    if thisBlock2Loop != None:
        for paramName in thisBlock2Loop:
            exec('{} = thisBlock2Loop[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    encodeLoop_Block2Run2 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='encodeLoop_Block2Run2')
    thisExp.addLoop(encodeLoop_Block2Run2)  # add the loop to the experiment
    thisEncodeLoop_Block2Run2 = encodeLoop_Block2Run2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block2Run2.rgb)
    if thisEncodeLoop_Block2Run2 != None:
        for paramName in thisEncodeLoop_Block2Run2:
            exec('{} = thisEncodeLoop_Block2Run2[paramName]'.format(paramName))
    
    for thisEncodeLoop_Block2Run2 in encodeLoop_Block2Run2:
        currentLoop = encodeLoop_Block2Run2
        # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block2Run2.rgb)
        if thisEncodeLoop_Block2Run2 != None:
            for paramName in thisEncodeLoop_Block2Run2:
                exec('{} = thisEncodeLoop_Block2Run2[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time) # Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 2 RUN 2 ENCODING
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'fixationText.started') # Original code based on internal clock (which can't be reset)
                fixation_start = next_fixCross_started_time if next_fixCross_started_time != 0 else Time_Since_Run.getTime()
                thisExp.addData('fixationText.started', fixation_start)  # fixationText.started for Block 2 Run 2
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 2 RUN 2 ENCODING
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'fixationText.stopped') # Original code based on internal clock (which can't be reset)
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 2 Run 2
                    fixationText.setAutoDraw(False)
                else:# If fixation_stop not updated
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop) # fixationText.stopped for Block 2 Run 2
                    
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 2 RUN 2 ENCODING
        
        # --- Prepare to start Routine "encodeTrial_Run2" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from encodeTrialRun2Code
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Random_Encode_List_Block_2_Run_2[current_encode_index]
        
        # Add the image_file data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_2_Run)
        
        encodeTrialTestKeyRespRun2.keys = []
        encodeTrialTestKeyRespRun2.rt = []
        _encodeTrialTestKeyRespRun2_allKeys = []
        # keep track of which components have finished
        encodeTrial_Run2Components = [encodeTrialRun2_Image, encodeTrialTestKeyRespRun2]
        for thisComponent in encodeTrial_Run2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encodeTrial_Run2" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from encodeTrialRun2Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
                
            
            
            # *encodeTrialRun2_Image* updates
            if encodeTrialRun2_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialRun2_Image.frameNStart = frameN  # exact frame index
                encodeTrialRun2_Image.tStart = t  # local t and not account for scr refresh
                encodeTrialRun2_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialRun2_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'encodeTrialRun2_Image.started') # Original code based on internal clock (which can't be reset)
                encodeImageRun2_started_time = fixation_stop  # Image onset at fixation cross stop time
                thisExp.addData('encodeTrialRun2_Image.started', encodeImageRun2_started_time)  # encodeTrialRun2_Image for Block 2 Run 2
                encodeTrialRun2_Image.setAutoDraw(True)
            if encodeTrialRun2_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialRun2_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialRun2_Image.tStop = t  # not accounting for scr refresh
                    encodeTrialRun2_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
#                    thisExp.timestampOnFlip(win, 'encodeTrialRun2_Image.stopped') # Original code based on internal clock (which can't be reset)
                    encodeImageRun2_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('encodeTrialRun2_Image.stopped', encodeImageRun2_stopped_time)  # encodeTrialRun2_Image.stopped for Block 2 Run 2
                    next_fixCross_started_time = encodeImageRun2_stopped_time  # Set onset of next fixation cross time
                    encodeTrialRun2_Image.setAutoDraw(False)
                else: #  If encodeImageRun2_stopped_time not updated
                    encodeImageRun2_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('encodeTrialRun2_Image.stopped', encodeImageRun2_stopped_time)  # encodeTrialRun2_Image.stopped for Block 2 Run 2 
                    next_fixCross_started_time = encodeImageRun2_stopped_time  # Set onset of next fixation cross time
            if encodeTrialRun2_Image.status == STARTED:  # only update if drawing
                encodeTrialRun2_Image.setImage(Random_Encode_List_Block_2_Run_2[current_encode_index], log=False)
            
            # *encodeTrialTestKeyRespRun2* updates
            waitOnFlip = False
            if encodeTrialTestKeyRespRun2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialTestKeyRespRun2.frameNStart = frameN  # exact frame index
                encodeTrialTestKeyRespRun2.tStart = t  # local t and not account for scr refresh
                encodeTrialTestKeyRespRun2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialTestKeyRespRun2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun2.started')  # Original code based on internal clock (which can't be reset)
                encodeTrialTestKeyRespRun2_started_time = encodeImageRun2_started_time  # Key Resp same as image onset/stop time
                thisExp.addData('encodeTrialTestKeyRespRun2.started', encodeTrialTestKeyRespRun2_started_time)  # encodeTrialTestKeyRespRun2.started for Block 2 Run 2
                encodeTrialTestKeyRespRun2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(encodeTrialTestKeyRespRun2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(encodeTrialTestKeyRespRun2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if encodeTrialTestKeyRespRun2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialTestKeyRespRun2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialTestKeyRespRun2.tStop = t  # not accounting for scr refresh
                    encodeTrialTestKeyRespRun2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
#                    thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun2.stopped') # Original code based on internal clock (which can't be reset)
                    encodeTrialTestKeyRespRun2_stopped_time = encodeImageRun2_stopped_time  # Key Resp same as image onset/stop time
                    thisExp.addData('encodeTrialTestKeyRespRun2.stopped', encodeTrialTestKeyRespRun2_stopped_time)  # encodeTrialTestKeyRespRun2.stopped for Block 2 Run 2
                    encodeTrialTestKeyRespRun2.status = FINISHED
                else: # If encodeTrialTestKeyRespRun2_stopped_time not updated
                    encodeTrialTestKeyRespRun2_stopped_time = encodeImageRun2_stopped_time  # Key Resp same as image onset/stop time
                    thisExp.addData('encodeTrialTestKeyRespRun2.stopped', encodeTrialTestKeyRespRun2_stopped_time)  # encodeTrialTestKeyRespRun2.stopped for Block 2 Run 2
            if encodeTrialTestKeyRespRun2.status == STARTED and not waitOnFlip:
                theseKeys = encodeTrialTestKeyRespRun2.getKeys(keyList=['1','2'], waitRelease=False)
                _encodeTrialTestKeyRespRun2_allKeys.extend(theseKeys)
                if len(_encodeTrialTestKeyRespRun2_allKeys):
                    encodeTrialTestKeyRespRun2.keys = _encodeTrialTestKeyRespRun2_allKeys[-1].name  # just the last key pressed
                    encodeTrialTestKeyRespRun2.rt = _encodeTrialTestKeyRespRun2_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encodeTrial_Run2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encodeTrial_Run2" ---
        for thisComponent in encodeTrial_Run2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from encodeTrialRun2Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        ## Update encoding lists for txt (Use for coder)
        update_encoding_data(
            block_num=2,
            run_num=Block_2_Run,
            random_encode_key=Random_Encode_Key_Block_2_Run_2,
            encode_image_started_time=encodeImageRun2_started_time,
            image_time=image_time,
            current_displayed_image=current_displayed_image,
            encoding_data_face=Encoding_Data_Face_Block_2_Run_2,
            encoding_data_place=Encoding_Data_Place_Block_2_Run_2,
            encoding_data_pair=Encoding_Data_Pair_Block_2_Run_2,
            encoding_response_data_face=Encoding_Response_Data_Face_Block_2_Run_2,
            encoding_response_data_place=Encoding_Response_Data_Place_Block_2_Run_2,
            encoding_response_data_pair=Encoding_Response_Data_Pair_Block_2_Run_2,
        )
        
        # Record encoding response
        gender_is_correct, liquid_is_correct, pair_fit_is_correct, image_gender, response_gender, image_water, response_water, response_pair_fit = update_encode_pre_recog_response_data(
            block_num=2,
            run_num=Block_2_Run,
            current_displayed_image=current_displayed_image,
            random_encode_key=Random_Encode_Key_Block_2_Run_2,
            response_key=encodeTrialTestKeyRespRun2.keys,
            liquid_conditions=["citybeach20", "beach1", "beach4"]
        )
        
        # Save encoding responses to output
        if gender_is_correct is not None:
            thisExp.addData('image_gender', image_gender)
            thisExp.addData('response_gender', response_gender)
            thisExp.addData('gender_is_correct', gender_is_correct)
        
        if liquid_is_correct is not None:
            thisExp.addData('image_water', image_water)
            thisExp.addData('response_water', response_water)
            thisExp.addData('liquid_is_correct', liquid_is_correct)
        
        if response_pair_fit is not None:
            thisExp.addData('response_pair_fit', response_pair_fit)
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block2Loop.thisRepN)
        
        # Increment the current encode index to iterate over images
        current_encode_index += 1
        
        if current_encode_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            encode_trials_counter += 1
        
        if current_encode_index == 12: # Last stimulus is shown
            
            current_encode_index = 0 # Reset stimulus counter
            current_encode_key_index += 1 # Next encode condition
            current_jitter_index = 0 # Reset current jitter index
            
            Encoding_Started = False # End of encoding 
            Fix_Cross_Started = False # Next ISI is instructions
            Recognition_Started = True # Start of recognition
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            if current_encode_key_index == 3: # At last encoding condition
                current_encode_key_index = 0 # Reset encode condition order
            else:
                Random_Encode_Key_Block_2_Run_2 = Encode_Key_List_Block_2_Run_2[current_encode_key_index]
                Random_Encode_List_Block_2_Run_2 = Encoding_Dictionary_Block_2_Run_2[Random_Encode_Key_Block_2_Run_2]
         
        # Iterate through jitter times if showing fixation cross and after first image
        if Fix_Cross_Started == True and current_encode_index > 1 : 
            current_jitter_index += 1
        # check responses
        if encodeTrialTestKeyRespRun2.keys in ['', [], None]:  # No response was made
            encodeTrialTestKeyRespRun2.keys = None
        encodeLoop_Block2Run2.addData('encodeTrialTestKeyRespRun2.keys',encodeTrialTestKeyRespRun2.keys)
        if encodeTrialTestKeyRespRun2.keys != None:  # we had a response
            encodeLoop_Block2Run2.addData('encodeTrialTestKeyRespRun2.rt', encodeTrialTestKeyRespRun2.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'encodeLoop_Block2Run2'
    
    
    # set up handler to look after randomisation of conditions etc
    recognitionLoop_Block2Run2 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='recognitionLoop_Block2Run2')
    thisExp.addLoop(recognitionLoop_Block2Run2)  # add the loop to the experiment
    thisRecognitionLoop_Block2Run2 = recognitionLoop_Block2Run2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block2Run2.rgb)
    if thisRecognitionLoop_Block2Run2 != None:
        for paramName in thisRecognitionLoop_Block2Run2:
            exec('{} = thisRecognitionLoop_Block2Run2[paramName]'.format(paramName))
    
    for thisRecognitionLoop_Block2Run2 in recognitionLoop_Block2Run2:
        currentLoop = recognitionLoop_Block2Run2
        # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block2Run2.rgb)
        if thisRecognitionLoop_Block2Run2 != None:
            for paramName in thisRecognitionLoop_Block2Run2:
                exec('{} = thisRecognitionLoop_Block2Run2[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time) # Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 2 RUN 2 RECOGNITION
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'fixationText.started') # Original code based on internal clock (which can't be reset)
                fixation_start = next_fixCross_started_time if next_fixCross_started_time != 0 else Time_Since_Run.getTime()
                thisExp.addData('fixationText.started', fixation_start)  # fixationText.started for Block 2 Run 2
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 2 RUN 2 RECOGNITION
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 2 Run 2
                    fixationText.setAutoDraw(False)
                else: # If fixation_stop not updated
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop) # fixationText.stopped for Block 2 Run 2
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 2 RUN 2 RECOGNITION
        
        # --- Prepare to start Routine "recognitionTrial_Run2" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from recognitonTrialRun2Code
        # Add the start time and volume counter to the data
        thisExp.addData('start_volume', volume_counter)
        
        # Reset the keyboard clock
        kb.clock.reset()  # when you want to start the timer from
        
        # Set the volume counter message
        volume_counter_message = str(volume_counter) + " out of " + str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Recog_List_Block_2_Run_2[current_recog_index]
        
        # Add the current_displayed_image to the data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_2_Run)
        
        recogTrialTestKeyRespRun2.keys = []
        recogTrialTestKeyRespRun2.rt = []
        _recogTrialTestKeyRespRun2_allKeys = []
        # keep track of which components have finished
        recognitionTrial_Run2Components = [recogTrialRun2_Image, recogTrialTestKeyRespRun2]
        for thisComponent in recognitionTrial_Run2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recognitionTrial_Run2" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from recognitonTrialRun2Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            

            
            # *recogTrialRun2_Image* updates
            if recogTrialRun2_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialRun2_Image.frameNStart = frameN  # exact frame index
                recogTrialRun2_Image.tStart = t  # local t and not account for scr refresh
                recogTrialRun2_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialRun2_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
#                thisExp.timestampOnFlip(win, 'recogTrialRun2_Image.started') # Original code based on internal clock (which can't be reset)
                recognitionImageRun2_started_time = fixation_stop
                thisExp.addData('recogTrialRun2_Image.started', recognitionImageRun2_started_time)  # recogTrialRun2_Image.started for Block 2 Run 2
                recogTrialRun2_Image.setAutoDraw(True)
            if recogTrialRun2_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialRun2_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialRun2_Image.tStop = t  # not accounting for scr refresh
                    recogTrialRun2_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #thisExp.timestampOnFlip(win, 'recogTrialRun2_Image.stopped') # Original code based on internal clock (which can't be reset)
                    recognitionImageRun2_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('recogTrialRun2_Image.stopped', recognitionImageRun2_stopped_time)  # recogTrialRun2_Image.stopped for Block 2 Run 2
                    next_fixCross_started_time = recognitionImageRun2_stopped_time
                    recogTrialRun2_Image.setAutoDraw(False)
                else: # If recognitionImageRun2_stopped_time not updated
                    recognitionImageRun2_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('recogTrialRun2_Image.stopped', recognitionImageRun2_stopped_time)  # recogTrialRun2_Image.stopped for Block 2 Run 2
                    next_fixCross_started_time = recognitionImageRun2_stopped_time
            if recogTrialRun2_Image.status == STARTED:  # only update if drawing
                recogTrialRun2_Image.setImage(Recog_List_Block_2_Run_2[current_recog_index], log=False)
            
            # *recogTrialTestKeyRespRun2* updates
            waitOnFlip = False
            if recogTrialTestKeyRespRun2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialTestKeyRespRun2.frameNStart = frameN  # exact frame index
                recogTrialTestKeyRespRun2.tStart = t  # local t and not account for scr refresh
                recogTrialTestKeyRespRun2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialTestKeyRespRun2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun2.started') # Original code based on internal clock (which can't be reset)
                recogTrialTestKeyRespRun2_started_time = recognitionImageRun2_started_time
                thisExp.addData('recogTrialTestKeyRespRun2.started', recogTrialTestKeyRespRun2_started_time)
                recogTrialTestKeyRespRun2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recogTrialTestKeyRespRun2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recogTrialTestKeyRespRun2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recogTrialTestKeyRespRun2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialTestKeyRespRun2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialTestKeyRespRun2.tStop = t  # not accounting for scr refresh
                    recogTrialTestKeyRespRun2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun2.stopped') # Original code based on internal clock (which can't be reset)
                    recogTrialTestKeyRespRun2_stopped_time = recognitionImageRun2_stopped_time
                    thisExp.addData('recogTrialTestKeyRespRun2.stopped', recogTrialTestKeyRespRun2_stopped_time)  # recogTrialTestKeyRespRun2.stopped for Block 2 Run 2
                    recogTrialTestKeyRespRun2.status = FINISHED
                else: # If recogTrialTestKeyRespRun2_started_time is not updated
                    recogTrialTestKeyRespRun2_stopped_time = recognitionImageRun2_stopped_time
                    thisExp.addData('recogTrialTestKeyRespRun2.stopped', recogTrialTestKeyRespRun2_stopped_time)  # recogTrialTestKeyRespRun2.stopped for Block 2 Run 2
            if recogTrialTestKeyRespRun2.status == STARTED and not waitOnFlip:
                theseKeys = recogTrialTestKeyRespRun2.getKeys(keyList=['1','2'], waitRelease=False)
                _recogTrialTestKeyRespRun2_allKeys.extend(theseKeys)
                if len(_recogTrialTestKeyRespRun2_allKeys):
                    recogTrialTestKeyRespRun2.keys = _recogTrialTestKeyRespRun2_allKeys[-1].name  # just the last key pressed
                    recogTrialTestKeyRespRun2.rt = _recogTrialTestKeyRespRun2_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recognitionTrial_Run2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recognitionTrial_Run2" ---
        for thisComponent in recognitionTrial_Run2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from recognitonTrialRun2Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        # Add to list for txt
        if Recog_Key_Block_2_Run_2 == 'Face_Recog_Key_Block_2_Run_2':
            Recog_Data_Face_Block_2_Run_2.append([recognitionImageRun2_started_time, image_time, 1.000])
        elif Recog_Key_Block_2_Run_2 == 'Place_Recog_Key_Block_2_Run_2':
            Recog_Data_Place_Block_2_Run_2.append([recognitionImageRun2_started_time, image_time, 1.000])
        elif Recog_Key_Block_2_Run_2 == 'Pair_Recog_Key_Block_2_Run_2':
            Recog_Data_Pair_Block_2_Run_2.append([recognitionImageRun2_started_time, image_time, 1.000])
        
        # Get the image OLD/NEW status and determine if the response is correct
        image_name = Recog_List_Block_2_Run_2[current_recog_index]
        if 'OLD' in image_name:
            image_old_new = "OLD"
        else:
            image_old_new = "NEW"
        # Add the image_old_new data to the experiment handler before response_old_new
        thisExp.addData('image_old_new', image_old_new)
        
        # Get the participant's response for the recognition trial
        if recogTrialTestKeyRespRun2.keys == '1':
            response_old_new = "OLD"
        elif recogTrialTestKeyRespRun2.keys == '2':
            response_old_new = "NEW"
        else:
            response_old_new = "NONE"
        # Add the response_old_new data to the experiment handler
        thisExp.addData('response_old_new', response_old_new)
        
        # Determine d prime and add results to the experiment handler
        is_correct = d_prime(image_old_new=image_old_new, response_key=recogTrialTestKeyRespRun2.keys)
        thisExp.addData('is_correct', is_correct)
        
        # Update encoding and recognition txt response list
        update_encoding_recognition_response_data(
            block_num=2, # Change based on block number
            run_num=Block_2_Run,  # Change based on run number
            is_correct=is_correct, 
            current_displayed_image=current_displayed_image, 
            recog_key=Recog_Key_Block_2_Run_2, # Change based on Block and condition
            image_start_time=recognitionImageRun2_started_time, # Change in coder to recognitionImageRun2_started_time
            image_time=image_time, 
            recog_response_data_face=Recog_Response_Data_Face_Block_2_Run_2,  # Change based on Block and condition
            recog_response_data_place=Recog_Response_Data_Place_Block_2_Run_2,  # Change based on Block and condition
            recog_response_data_pair=Recog_Response_Data_Pair_Block_2_Run_2,  # Change based on Block and condition
            encoding_response_data_face=Encoding_Response_Data_Face_Block_2_Run_2,  # Change based on Block and condition
            encoding_response_data_place=Encoding_Response_Data_Place_Block_2_Run_2,  # Change based on Block and condition
            encoding_response_data_pair=Encoding_Response_Data_Pair_Block_2_Run_2  # Change based on Block and condition
        )
        
        # Increment the current recognition index to iterate over images
        current_recog_index += 1
        
        if current_recog_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            recognition_trials_counter += 1
            
        if current_recog_index == 12: # Last stimulus is shown
            # Increment the number of trialis in block loop
            block_loop_trials_counter += 1
            
            Recognition_Started = False # End of recognition
            Fix_Cross_Started = False # Next ISI is instructions
            Encoding_Started = True # Start of encoding 
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            # Reset the current recognition index 
            current_recog_index = 0
            
            # Reset current jitter index
            current_jitter_index = 0
            
            # Determine the corresponding recognition key based on the chosen encoding key
            if Random_Encode_Key_Block_2_Run_2  == 'Face_Encode_Key_Block_2_Run_2':
                Recog_Key_Block_2_Run_2 = 'Face_Recog_Key_Block_2_Run_2'
            elif Random_Encode_Key_Block_2_Run_2 == 'Place_Encode_Key_Block_2_Run_2':
                Recog_Key_Block_2_Run_2 = 'Place_Recog_Key_Block_2_Run_2'
            elif Random_Encode_Key_Block_2_Run_2 == 'Pair_Encode_Key_Block_2_Run_2':
                Recog_Key_Block_2_Run_2 = 'Pair_Recog_Key_Block_2_Run_2'
        
            # Get the corresponding recognition list based on the determined recognition key
            Recog_List_Block_2_Run_2 = Recognition_Dictionary_Block_2_Run_2[Recog_Key_Block_2_Run_2]
        
        # Iterate through jitter times if showing fixation cross and after first image
        if Fix_Cross_Started == True and current_recog_index > 1 : 
            current_jitter_index += 1
         
        # Reset the trial counters
        if block_loop_trials_counter == 3:
            block_loop_trials_counter = 0
            recognition_trials_counter = 0
            encode_trials_counter = 0
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block2Loop.thisRepN)
        
        # When Block Loop ends based on stimuli total
        if len(Block_Loop_Voulme_Counter) == 72: 
            volume_counter = 1 # Reset volume_counter
            Block_Loop_Voulme_Counter = [] # Reinitalize counter
            Block_2_Run_2_Started = False # Block 2 Run 2 finished
            Encoding_Started = True # Start of encoding for next block 
            Recognition_Started = False # End of recognition for next block
            Block_3_Run_3_Started = True # Begin next block
        
        # check responses
        if recogTrialTestKeyRespRun2.keys in ['', [], None]:  # No response was made
            recogTrialTestKeyRespRun2.keys = None
        recognitionLoop_Block2Run2.addData('recogTrialTestKeyRespRun2.keys',recogTrialTestKeyRespRun2.keys)
        if recogTrialTestKeyRespRun2.keys != None:  # we had a response
            recognitionLoop_Block2Run2.addData('recogTrialTestKeyRespRun2.rt', recogTrialTestKeyRespRun2.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'recognitionLoop_Block2Run2'
    
    thisExp.nextEntry()
    
# completed 3.0 repeats of 'Block2Loop'


# --- Prepare to start Routine "triggerSync" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# Run 'Begin Routine' code from triggerSyncCode
trigger_sync_message = "+"
if volume_counter == 1:
    trigger_sync_message = "Waiting for scanner..."




triggerSyncText.setText(trigger_sync_message)
triggerSync_Key_Resp.keys = []
triggerSync_Key_Resp.rt = []
_triggerSync_Key_Resp_allKeys = []
# keep track of which components have finished
triggerSyncComponents = [triggerSyncText, triggerSync_Key_Resp]
for thisComponent in triggerSyncComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1
Time_Since_Run.reset()  # Reset for Block 3 Run 3
next_fixCross_started_time = 0  # Reset for Block 3 Run 3

# --- Run Routine "triggerSync" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *triggerSyncText* updates
    if triggerSyncText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSyncText.frameNStart = frameN  # exact frame index
        triggerSyncText.tStart = t  # local t and not account for scr refresh
        triggerSyncText.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSyncText, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        #        thisExp.timestampOnFlip(win, 'triggerSyncText.started') # Original code based on internal clock (which can't be reset)
        triggerSync_started_time = Time_Since_Run.getTime()
        thisExp.addData('triggerSyncText.started', triggerSync_started_time)  # triggerSyncText.started for Block 3 Run 3
        triggerSyncText.setAutoDraw(True)
    
    # *triggerSync_Key_Resp* updates
    waitOnFlip = False
    if triggerSync_Key_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        triggerSync_Key_Resp.frameNStart = frameN  # exact frame index
        triggerSync_Key_Resp.tStart = t  # local t and not account for scr refresh
        triggerSync_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(triggerSync_Key_Resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        #        thisExp.timestampOnFlip(win, 'triggerSync_Key_Resp.started') # Original code based on internal clock (which can't be reset)
        thisExp.addData('triggerSync_Key_Resp.started', triggerSync_started_time)  # triggerSyncKeyResp.started for Block 3 Run 3
        triggerSync_Key_Resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(triggerSync_Key_Resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(triggerSync_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if triggerSync_Key_Resp.status == STARTED and not waitOnFlip:
        theseKeys = triggerSync_Key_Resp.getKeys(keyList=['5','t','s'], waitRelease=False)
        _triggerSync_Key_Resp_allKeys.extend(theseKeys)
        if len(_triggerSync_Key_Resp_allKeys):
            triggerSync_Key_Resp.keys = _triggerSync_Key_Resp_allKeys[-1].name  # just the last key pressed
            triggerSync_Key_Resp.rt = _triggerSync_Key_Resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in triggerSyncComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "triggerSync" ---
for thisComponent in triggerSyncComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if triggerSync_Key_Resp.keys in ['', [], None]:  # No response was made
    triggerSync_Key_Resp.keys = None
thisExp.addData('triggerSync_Key_Resp.keys',triggerSync_Key_Resp.keys)
if triggerSync_Key_Resp.keys != None:  # we had a response
    thisExp.addData('triggerSync_Key_Resp.rt', triggerSync_Key_Resp.rt)
thisExp.nextEntry()
# the Routine "triggerSync" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Block3Loop = data.TrialHandler(nReps=3.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Block3Loop')
thisExp.addLoop(Block3Loop)  # add the loop to the experiment
thisBlock3Loop = Block3Loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock3Loop.rgb)
if thisBlock3Loop != None:
    for paramName in thisBlock3Loop:
        exec('{} = thisBlock3Loop[paramName]'.format(paramName))

for thisBlock3Loop in Block3Loop:
    currentLoop = Block3Loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlock3Loop.rgb)
    if thisBlock3Loop != None:
        for paramName in thisBlock3Loop:
            exec('{} = thisBlock3Loop[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    encodeLoop_Block3Run3 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='encodeLoop_Block3Run3')
    thisExp.addLoop(encodeLoop_Block3Run3)  # add the loop to the experiment
    thisEncodeLoop_Block3Run3 = encodeLoop_Block3Run3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block3Run3.rgb)
    if thisEncodeLoop_Block3Run3 != None:
        for paramName in thisEncodeLoop_Block3Run3:
            exec('{} = thisEncodeLoop_Block3Run3[paramName]'.format(paramName))
    
    for thisEncodeLoop_Block3Run3 in encodeLoop_Block3Run3:
        currentLoop = encodeLoop_Block3Run3
        # abbreviate parameter names if possible (e.g. rgb = thisEncodeLoop_Block3Run3.rgb)
        if thisEncodeLoop_Block3Run3 != None:
            for paramName in thisEncodeLoop_Block3Run3:
                exec('{} = thisEncodeLoop_Block3Run3[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time) # Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 3 RUN 3 ENCODING
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'fixationText.started') # Original code based on internal clock (which can't be reset)
                fixation_start = next_fixCross_started_time if next_fixCross_started_time != 0 else Time_Since_Run.getTime()
                thisExp.addData('fixationText.started', fixation_start)  # fixationText.started for Block 3 Run 3
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 3 RUN 3 ENCODING
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'fixationText.stopped') # Original code based on internal clock (which can't be reset)
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 3 Run 3
                    fixationText.setAutoDraw(False)
                else:# If fixation_stop not updated
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 3 Run 3
                    
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 3 RUN 3 ENCODING
        
        # --- Prepare to start Routine "encodeTrial_Run3" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from encodeTrialRun3Code
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Random_Encode_List_Block_3_Run_3[current_encode_index]
        
        # Add the image_file data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_3_Run)
        
        encodeTrialTestKeyRespRun3.keys = []
        encodeTrialTestKeyRespRun3.rt = []
        _encodeTrialTestKeyRespRun3_allKeys = []
        # keep track of which components have finished
        encodeTrial_Run3Components = [encodeTrialRun3_Image, encodeTrialTestKeyRespRun3]
        for thisComponent in encodeTrial_Run3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encodeTrial_Run3" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from encodeTrialRun3Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
                
                        
            # *encodeTrialRun3_Image* updates
            if encodeTrialRun3_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialRun3_Image.frameNStart = frameN  # exact frame index
                encodeTrialRun3_Image.tStart = t  # local t and not account for scr refresh
                encodeTrialRun3_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialRun3_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'encodeTrialRun3_Image.started') # Original code based on internal clock (which can't be reset)
                encodeImageRun3_started_time = fixation_stop
                thisExp.addData('encodeTrialRun3_Image.started', encodeImageRun3_started_time)  # encodeTrialRun3_Image.started for Block 3 Run 3
                encodeTrialRun3_Image.setAutoDraw(True)
            if encodeTrialRun3_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialRun3_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialRun3_Image.tStop = t  # not accounting for scr refresh
                    encodeTrialRun3_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'encodeTrialRun3_Image.stopped')  # Original code based on internal clock (which can't be reset)
                    encodeImageRun3_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('encodeTrialRun3_Image.stopped', encodeImageRun3_stopped_time)  # encodeTrialRun3_Image.stopped for Block 3 Run 3
                    next_fixCross_started_time = encodeImageRun3_stopped_time
                    encodeTrialRun3_Image.setAutoDraw(False)
                else: # If encodeImageRun3_stopped_time not updated
                    encodeImageRun3_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('encodeTrialRun3_Image.stopped', encodeImageRun3_stopped_time)  # encodeTrialRun3_Image.stopped for Block 3 Run 3
                    next_fixCross_started_time = encodeImageRun3_stopped_time
            if encodeTrialRun3_Image.status == STARTED:  # only update if drawing
                encodeTrialRun3_Image.setImage(Random_Encode_List_Block_3_Run_3[current_encode_index], log=False)
            
            # *encodeTrialTestKeyRespRun3* updates
            waitOnFlip = False
            if encodeTrialTestKeyRespRun3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                encodeTrialTestKeyRespRun3.frameNStart = frameN  # exact frame index
                encodeTrialTestKeyRespRun3.tStart = t  # local t and not account for scr refresh
                encodeTrialTestKeyRespRun3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encodeTrialTestKeyRespRun3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun3.started') # Original code based on internal clock (which can't be reset)
                encodeTrialTestKeyRespRun3_started_time = encodeImageRun3_started_time
                thisExp.addData('encodeTrialTestKeyRespRun3.started', encodeTrialTestKeyRespRun3_started_time)  # encodeTrialTestKeyRespRun3.started for Block 3 Run 3
                encodeTrialTestKeyRespRun3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(encodeTrialTestKeyRespRun3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(encodeTrialTestKeyRespRun3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if encodeTrialTestKeyRespRun3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encodeTrialTestKeyRespRun3.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    encodeTrialTestKeyRespRun3.tStop = t  # not accounting for scr refresh
                    encodeTrialTestKeyRespRun3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'encodeTrialTestKeyRespRun3.stopped') # Original code based on internal clock (which can't be reset)
                    encodeTrialTestKeyRespRun3_stopped_time = encodeImageRun3_stopped_time
                    thisExp.addData('encodeTrialTestKeyRespRun3.stopped', encodeTrialTestKeyRespRun3_stopped_time)  # encodeTrialTestKeyRespRun3.stopped for Block 3 Run 3
                    encodeTrialTestKeyRespRun3.status = FINISHED
                else: # If encodeTrialTestKeyRespRun3_stopped_time not updated
                    encodeTrialTestKeyRespRun3_stopped_time = encodeImageRun3_stopped_time
                    thisExp.addData('encodeTrialTestKeyRespRun3.stopped', encodeTrialTestKeyRespRun3_stopped_time)  # encodeTrialTestKeyRespRun3.stopped for Block 3 Run 3
            if encodeTrialTestKeyRespRun3.status == STARTED and not waitOnFlip:
                theseKeys = encodeTrialTestKeyRespRun3.getKeys(keyList=['1','2'], waitRelease=False)
                _encodeTrialTestKeyRespRun3_allKeys.extend(theseKeys)
                if len(_encodeTrialTestKeyRespRun3_allKeys):
                    encodeTrialTestKeyRespRun3.keys = _encodeTrialTestKeyRespRun3_allKeys[-1].name  # just the last key pressed
                    encodeTrialTestKeyRespRun3.rt = _encodeTrialTestKeyRespRun3_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encodeTrial_Run3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encodeTrial_Run3" ---
        for thisComponent in encodeTrial_Run3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from encodeTrialRun3Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        ## Update encoding lists for txt (Use for coder)
        update_encoding_data(
            block_num=3,
            run_num=Block_3_Run,
            random_encode_key=Random_Encode_Key_Block_3_Run_3,
            encode_image_started_time=encodeImageRun3_started_time,
            image_time=image_time,
            current_displayed_image=current_displayed_image,
            encoding_data_face=Encoding_Data_Face_Block_3_Run_3,
            encoding_data_place=Encoding_Data_Place_Block_3_Run_3,
            encoding_data_pair=Encoding_Data_Pair_Block_3_Run_3,
            encoding_response_data_face=Encoding_Response_Data_Face_Block_3_Run_3,
            encoding_response_data_place=Encoding_Response_Data_Place_Block_3_Run_3,
            encoding_response_data_pair=Encoding_Response_Data_Pair_Block_3_Run_3,
        )
        
        # Record encoding response
        gender_is_correct, liquid_is_correct, pair_fit_is_correct, image_gender, response_gender, image_water, response_water, response_pair_fit = update_encode_pre_recog_response_data(
            block_num=3,
            run_num=Block_3_Run,
            current_displayed_image=current_displayed_image,
            random_encode_key=Random_Encode_Key_Block_3_Run_3,
            response_key=encodeTrialTestKeyRespRun3.keys,
            liquid_conditions=["EMBfemale21-3_w_beach13_OLD", "EMBfemale21-3_w_beach14_OLD", "EMBfemale21-3_w_beach15_OLD", "EMBfemale21-3_w_city4_OLD", "EMBfemale21-3_w_hills4_OLD"]
        )
        
        # Save encoding responses to output
        if gender_is_correct is not None:
            thisExp.addData('image_gender', image_gender)
            thisExp.addData('response_gender', response_gender)
            thisExp.addData('gender_is_correct', gender_is_correct)
        
        if liquid_is_correct is not None:
            thisExp.addData('image_water', image_water)
            thisExp.addData('response_water', response_water)
            thisExp.addData('liquid_is_correct', liquid_is_correct)
        
        if response_pair_fit is not None:
            thisExp.addData('response_pair_fit', response_pair_fit)
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block3Loop.thisRepN)
        
        # Increment the current encode index to iterate over images
        current_encode_index += 1
        
        if current_encode_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            encode_trials_counter += 1
        
        if current_encode_index == 12: # Last stimulus is shown
            
            current_encode_index = 0 # Reset stimulus counter
            current_encode_key_index += 1 # Next encode condition
            current_jitter_index = 0 # Reset current jitter index
            
            Encoding_Started = False # End of encoding 
            Fix_Cross_Started = False # Next ISI is instructions
            Recognition_Started = True # Start of recognition
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            if current_encode_key_index == 3: # At last encoding condition
                current_encode_key_index = 0 # Reset encode condition order
            else:
                Random_Encode_Key_Block_3_Run_3 = Encode_Key_List_Block_3_Run_3[current_encode_key_index]
                Random_Encode_List_Block_3_Run_3 = Encoding_Dictionary_Block_3_Run_3[Random_Encode_Key_Block_3_Run_3]
         
        # Iterate through jitter times if showing fixation cross and after first image
        if Fix_Cross_Started == True and current_encode_index > 1 : 
            current_jitter_index += 1
        
        # check responses
        if encodeTrialTestKeyRespRun3.keys in ['', [], None]:  # No response was made
            encodeTrialTestKeyRespRun3.keys = None
        encodeLoop_Block3Run3.addData('encodeTrialTestKeyRespRun3.keys',encodeTrialTestKeyRespRun3.keys)
        if encodeTrialTestKeyRespRun3.keys != None:  # we had a response
            encodeLoop_Block3Run3.addData('encodeTrialTestKeyRespRun3.rt', encodeTrialTestKeyRespRun3.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'encodeLoop_Block3Run3'
    
    
    # set up handler to look after randomisation of conditions etc
    recognitionLoop_Block3Run3 = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='recognitionLoop_Block3Run3')
    thisExp.addLoop(recognitionLoop_Block3Run3)  # add the loop to the experiment
    thisRecognitionLoop_Block3Run3 = recognitionLoop_Block3Run3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block3Run3.rgb)
    if thisRecognitionLoop_Block3Run3 != None:
        for paramName in thisRecognitionLoop_Block3Run3:
            exec('{} = thisRecognitionLoop_Block3Run3[paramName]'.format(paramName))
    
    for thisRecognitionLoop_Block3Run3 in recognitionLoop_Block3Run3:
        currentLoop = recognitionLoop_Block3Run3
        # abbreviate parameter names if possible (e.g. rgb = thisRecognitionLoop_Block3Run3.rgb)
        if thisRecognitionLoop_Block3Run3 != None:
            for paramName in thisRecognitionLoop_Block3Run3:
                exec('{} = thisRecognitionLoop_Block3Run3[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "fixationCross" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from fixationCode
        if(volume_counter == 1):
            start_time = timer.getTime();
        
        # Set ISI time and display
        ISI_Time, ISI_Message = set_ISI_time_and_message(Fix_Cross_Started, jitter_duration, current_jitter_index,
                                                         Block_1_Run_1_Started, Block_2_Run_2_Started, Block_3_Run_3_Started,
                                                         Encoding_Started, Recognition_Started,
                                                         Random_Encode_Key_Block_1_Run_1, Recog_Key_Block_1_Run_1,
                                                         Random_Encode_Key_Block_2_Run_2, Recog_Key_Block_2_Run_2,
                                                         Random_Encode_Key_Block_3_Run_3, Recog_Key_Block_3_Run_3)
#        print(ISI_Time) # Uncomment to print ISI TIME
        thisExp.addData('jitter_time',ISI_Time)
        current_time = timer.getTime() - start_time
        thisExp.addData('start_time',current_time)
        
        thisExp.addData('start_volume',volume_counter)
        kb.clock.reset()  # when you want to start the timer from
        volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
        # keep track of which components have finished
        fixationCrossComponents = [fixationText]
        for thisComponent in fixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixationCross" ---
        while continueRoutine and routineTimer.getTime() < ISI_Time:  # REPLACED 9999.0 FOR BLOCK 3 RUN 3 RECOGNITION
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from fixationCode
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            # *fixationText* updates
            if fixationText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationText.frameNStart = frameN  # exact frame index
                fixationText.tStart = t  # local t and not account for scr refresh
                fixationText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'fixationText.started') # Original code based on internal clock (which can't be reset)
                fixation_start = next_fixCross_started_time if next_fixCross_started_time != 0 else Time_Since_Run.getTime()
                thisExp.addData('fixationText.started', fixation_start)  # fixationText.started for Block 3 Run 3
                fixationText.setAutoDraw(True)
            if fixationText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationText.tStartRefresh + ISI_Time-frameTolerance: # REPLACED 9999 FOR BLOCK 3 RUN 3 RECOGNITION
                    # keep track of stop time/frame for later
                    fixationText.tStop = t  # not accounting for scr refresh
                    fixationText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'fixationText.stopped')  # Original code based on internal clock (which can't be reset)
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 3 Run 3
                    fixationText.setAutoDraw(False)
                else: # If fixation_stop not updated
                    fixation_stop = Time_Since_Run.getTime()
                    thisExp.addData('fixationText.stopped', fixation_stop)  # fixationText.stopped for Block 3 Run 3
            if fixationText.status == STARTED:  # only update if drawing
                fixationText.setText(ISI_Message, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationCross" ---
        for thisComponent in fixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-ISI_Time)  # REPLACED -9999.000000 FOR BLOCK 3 RUN 3 RECOGNITION
        
        # --- Prepare to start Routine "recognitionTrial_Run3" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from recognitonTrialRun3Code
        # Add the start time and volume counter to the data
        thisExp.addData('start_volume', volume_counter)
        
        # Reset the keyboard clock
        kb.clock.reset()  # when you want to start the timer from
        
        # Set the volume counter message
        volume_counter_message = str(volume_counter) + " out of " + str(volume_total)
        
        current_time_key = 0
        
        # Update the current_displayed_image variable
        current_displayed_image = Recog_List_Block_3_Run_3[current_recog_index]
        
        # Add the current_displayed_image to the data
        thisExp.addData('image_file', current_displayed_image)
        
        # Add the run to the data
        thisExp.addData('run', Block_3_Run)
        
        recogTrialTestKeyRespRun3.keys = []
        recogTrialTestKeyRespRun3.rt = []
        _recogTrialTestKeyRespRun3_allKeys = []
        # keep track of which components have finished
        recognitionTrial_Run3Components = [recogTrialRun3_Image, recogTrialTestKeyRespRun3]
        for thisComponent in recognitionTrial_Run3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recognitionTrial_Run3" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from recognitonTrialRun3Code
            # during your trial
            keys = kb.getKeys(['5','t','s'], waitRelease=True)
            for key in keys:
                if volume_counter <= 1:
                    start_time_key = timer_key.getTime()
                    current_time_key = timer_key.getTime() - start_time_key
                else:
                    current_time_key = timer_key.getTime() - start_time_key
                    start_time_key = timer_key.getTime()
                
                thisExp.addData('trigger_time', current_time_key)
                
                thisExp.addData('end_volume',volume_counter)
                volume_counter_message = str(volume_counter)+" out of "+str(volume_total)
                volume_counter += 1
                
            
            
            
            # *recogTrialRun3_Image* updates
            if recogTrialRun3_Image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialRun3_Image.frameNStart = frameN  # exact frame index
                recogTrialRun3_Image.tStart = t  # local t and not account for scr refresh
                recogTrialRun3_Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialRun3_Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'recogTrialRun3_Image.started') # Original code based on internal clock (which can't be reset)
                recognitionImageRun3_started_time = fixation_stop
                thisExp.addData('recogTrialRun3_Image.started', recognitionImageRun3_started_time)  # recogTrialRun3_Image.started for Block 3 Run 3
                recogTrialRun3_Image.setAutoDraw(True)
            if recogTrialRun3_Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialRun3_Image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialRun3_Image.tStop = t  # not accounting for scr refresh
                    recogTrialRun3_Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'recogTrialRun3_Image.stopped') # Original code based on internal clock (which can't be reset)
                    recognitionImageRun3_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('recogTrialRun3_Image.stopped', recognitionImageRun3_stopped_time)  # recogTrialRun3_Image.stopped for Block 3 Run 3
                    next_fixCross_started_time = recognitionImageRun3_stopped_time
                    recogTrialRun3_Image.setAutoDraw(False)
                else:  # If recognitionImageRun3_stopped_time not updated
                    recognitionImageRun3_stopped_time = Time_Since_Run.getTime()
                    thisExp.addData('recogTrialRun3_Image.stopped', recognitionImageRun3_stopped_time)  # recogTrialRun3_Image.stopped for Block 3 Run 3
                    next_fixCross_started_time = recognitionImageRun3_stopped_time
            if recogTrialRun3_Image.status == STARTED:  # only update if drawing
                recogTrialRun3_Image.setImage(Recog_List_Block_3_Run_3[current_recog_index], log=False)
            
            # *recogTrialTestKeyRespRun3* updates
            waitOnFlip = False
            if recogTrialTestKeyRespRun3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                recogTrialTestKeyRespRun3.frameNStart = frameN  # exact frame index
                recogTrialTestKeyRespRun3.tStart = t  # local t and not account for scr refresh
                recogTrialTestKeyRespRun3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recogTrialTestKeyRespRun3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                #                thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun3.started') # Original code based on internal clock (which can't be reset)
                recogTrialTestKeyRespRun3_started_time = recognitionImageRun3_started_time
                thisExp.addData('recogTrialTestKeyRespRun3.started', recogTrialTestKeyRespRun3_started_time)  # recogTrialTestKeyRespRun3.started for Block 3 Run 3
                recogTrialTestKeyRespRun3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recogTrialTestKeyRespRun3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recogTrialTestKeyRespRun3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recogTrialTestKeyRespRun3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recogTrialTestKeyRespRun3.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    recogTrialTestKeyRespRun3.tStop = t  # not accounting for scr refresh
                    recogTrialTestKeyRespRun3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    #                    thisExp.timestampOnFlip(win, 'recogTrialTestKeyRespRun3.stopped') # Original code based on internal clock (which can't be reset)
                    recogTrialTestKeyRespRun3_stopped_time = recognitionImageRun3_stopped_time
                    thisExp.addData('recogTrialTestKeyRespRun3.stopped', recogTrialTestKeyRespRun3_stopped_time) # recogTrialTestKeyRespRun3.stopped for Block 3 Run 3
                    recogTrialTestKeyRespRun3.status = FINISHED
                else: # If recogTrialTestKeyRespRun3_stopped_time not updated
                    recogTrialTestKeyRespRun3_stopped_time = recognitionImageRun3_stopped_time
                    thisExp.addData('recogTrialTestKeyRespRun3.stopped', recogTrialTestKeyRespRun3_stopped_time)
            if recogTrialTestKeyRespRun3.status == STARTED and not waitOnFlip:
                theseKeys = recogTrialTestKeyRespRun3.getKeys(keyList=['1','2'], waitRelease=False)
                _recogTrialTestKeyRespRun3_allKeys.extend(theseKeys)
                if len(_recogTrialTestKeyRespRun3_allKeys):
                    recogTrialTestKeyRespRun3.keys = _recogTrialTestKeyRespRun3_allKeys[-1].name  # just the last key pressed
                    recogTrialTestKeyRespRun3.rt = _recogTrialTestKeyRespRun3_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recognitionTrial_Run3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recognitionTrial_Run3" ---
        for thisComponent in recognitionTrial_Run3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from recognitonTrialRun3Code
        # Calculate and add the end time to the data
        current_time = timer.getTime() - start_time
        thisExp.addData('end_time', current_time)
        
        # Get the image OLD/NEW status and determine if the response is correct
        image_name = Recog_List_Block_3_Run_3[current_recog_index]
        if 'OLD' in image_name:
            image_old_new = "OLD"
        else:
            image_old_new = "NEW"
        # Add the image_old_new data to the experiment handler before response_old_new
        thisExp.addData('image_old_new', image_old_new)
        
        ## Add to list for txt
        if Recog_Key_Block_3_Run_3 == 'Face_Recog_Key_Block_3_Run_3':
            Recog_Data_Face_Block_3_Run_3.append([recognitionImageRun3_started_time, image_time, 1.000])
        elif Recog_Key_Block_3_Run_3 == 'Place_Recog_Key_Block_3_Run_3':
            Recog_Data_Place_Block_3_Run_3.append([recognitionImageRun3_started_time, image_time, 1.000])
        elif Recog_Key_Block_3_Run_3 == 'Pair_Recog_Key_Block_3_Run_3':
            Recog_Data_Pair_Block_3_Run_3.append([recognitionImageRun3_started_time, image_time, 1.000])
        
        # Get the participant's response for the recognition trial
        if recogTrialTestKeyRespRun3.keys == '1':
            response_old_new = "OLD"
        elif recogTrialTestKeyRespRun3.keys == '2':
            response_old_new = "NEW"
        else:
            response_old_new = "NONE"
        # Add the response_old_new data to the experiment handler
        thisExp.addData('response_old_new', response_old_new)
        
        # Determine d prime and add results to the experiment handler
        is_correct = d_prime(image_old_new=image_old_new, response_key=recogTrialTestKeyRespRun3.keys)
        thisExp.addData('is_correct', is_correct)
        
        # Update encoding and recognition txt response list
        update_encoding_recognition_response_data(
            block_num=3, # Change based on block number
            run_num=Block_3_Run,  # Change based on run number
            is_correct=is_correct, 
            current_displayed_image=current_displayed_image, 
            recog_key=Recog_Key_Block_3_Run_3, # Change based on Block and condition
            image_start_time=recognitionImageRun3_started_time, # Change in coder to recognitionImageRun3_started_time
            image_time=image_time, 
            recog_response_data_face=Recog_Response_Data_Face_Block_3_Run_3,  # Change based on Block and condition
            recog_response_data_place=Recog_Response_Data_Place_Block_3_Run_3,  # Change based on Block and condition
            recog_response_data_pair=Recog_Response_Data_Pair_Block_3_Run_3,  # Change based on Block and condition
            encoding_response_data_face=Encoding_Response_Data_Face_Block_3_Run_3,  # Change based on Block and condition
            encoding_response_data_place=Encoding_Response_Data_Place_Block_3_Run_3,  # Change based on Block and condition
            encoding_response_data_pair=Encoding_Response_Data_Pair_Block_3_Run_3  # Change based on Block and condition
        )
        
        # Increment the current recognition index to iterate over images
        current_recog_index += 1
        
        if current_recog_index == 1: # First stimulus is shown
            Fix_Cross_Started = True # Next ISI is '+'
            recognition_trials_counter += 1
            
        if current_recog_index == 12: # Last stimulus is shown
            # Increment the number of trialis in block loop
            block_loop_trials_counter += 1
            
            Recognition_Started = False # End of recognition
            Fix_Cross_Started = False # Next ISI is instructions
            Encoding_Started = True # Start of encoding 
            
            # Generate new random jitter times for next condition
            jitter_duration = generate_jitter_time(min_=5, max_=7, count=11, target=66.0)
            
            # Reset the current recognition index 
            current_recog_index = 0
            
            # Reset current jitter index
            current_jitter_index = 0
            
            # Determine the corresponding recognition key based on the chosen encoding key
            if Random_Encode_Key_Block_3_Run_3  == 'Face_Encode_Key_Block_3_Run_3':
                Recog_Key_Block_3_Run_3 = 'Face_Recog_Key_Block_3_Run_3'
            elif Random_Encode_Key_Block_3_Run_3 == 'Place_Encode_Key_Block_3_Run_3':
                Recog_Key_Block_3_Run_3 = 'Place_Recog_Key_Block_3_Run_3'
            elif Random_Encode_Key_Block_3_Run_3 == 'Pair_Encode_Key_Block_3_Run_3':
                Recog_Key_Block_3_Run_3 = 'Pair_Recog_Key_Block_3_Run_3'
        
            # Get the corresponding recognition list based on the determined recognition key
            Recog_List_Block_3_Run_3 = Recognition_Dictionary_Block_3_Run_3[Recog_Key_Block_3_Run_3]
        
        # Iterate through jitter times if showing fixation cross and after first image
        if Fix_Cross_Started == True and current_recog_index > 1 : 
            current_jitter_index += 1
         
        # Reset the trial counters
        if block_loop_trials_counter == 3:
            block_loop_trials_counter = 0
            recognition_trials_counter = 0
            encode_trials_counter = 0
        
        # For resetting volume_counter based on Block loop iteration 
        Block_Loop_Voulme_Counter.append(Block3Loop.thisRepN)
        
        # When Block Loop ends based on stimuli total
        if len(Block_Loop_Voulme_Counter) == 72: 
            volume_counter = 1 # Reset volume_counter
            Block_Loop_Voulme_Counter = [] # Reinitalize counter
            Block_3_Run_3_Started = False # Block 3 Run 3 finished
            Encoding_Started = True 
            Recognition_Started = False 
            # END OF EXPERIMENT
        
        # check responses
        if recogTrialTestKeyRespRun3.keys in ['', [], None]:  # No response was made
            recogTrialTestKeyRespRun3.keys = None
        recognitionLoop_Block3Run3.addData('recogTrialTestKeyRespRun3.keys',recogTrialTestKeyRespRun3.keys)
        if recogTrialTestKeyRespRun3.keys != None:  # we had a response
            recognitionLoop_Block3Run3.addData('recogTrialTestKeyRespRun3.rt', recogTrialTestKeyRespRun3.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 12.0 repeats of 'recognitionLoop_Block3Run3'
    
    thisExp.nextEntry()
    
# completed 3.0 repeats of 'Block3Loop'


# Run 'End Experiment' code from recognitonTrialRun3Code
# Create the output directory if it doesn't exist
output_directory = 'output_directory'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get the participant number from the experiment info
participant_number = expInfo['participant']

# BLOCK 1 RUN 1
# Save the encoding and recognition data to a file for txt seperately
save_three_column_file(Encoding_Data_Face_Block_1_Run_1, os.path.join(output_directory, f'encoding_data_face_run1_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Place_Block_1_Run_1, os.path.join(output_directory, f'encoding_data_place_run1_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Pair_Block_1_Run_1, os.path.join(output_directory, f'encoding_data_pair_run1_P{participant_number}.txt'))

save_three_column_file(Recog_Data_Face_Block_1_Run_1, os.path.join(output_directory, f'recog_data_face_block1_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Place_Block_1_Run_1, os.path.join(output_directory, f'recog_data_place_block1_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Pair_Block_1_Run_1, os.path.join(output_directory, f'recog_data_pair_block1_P{participant_number}.txt'))

# Convert the remaining encoding and recognition response data in the dictionaries to lists and save them to text files
save_three_column_file(list(Encoding_Response_Data_Face_Block_1_Run_1.values()), os.path.join(output_directory, f'encoding_response_data_face_run1_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Place_Block_1_Run_1.values()), os.path.join(output_directory, f'encoding_response_data_place_run1_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Pair_Block_1_Run_1.values()), os.path.join(output_directory, f'encoding_response_data_pair_run1_P{participant_number}.txt'))

# Save the recognition data in the lists to text files
save_three_column_file(Recog_Response_Data_Face_Block_1_Run_1, os.path.join(output_directory, f'recog_response_data_face_run1_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Place_Block_1_Run_1, os.path.join(output_directory, f'recog_response_data_place_run1_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Pair_Block_1_Run_1, os.path.join(output_directory, f'recog_response_data_pair_run1_P{participant_number}.txt'))

# BLOCK 2 RUN 2
# Save the encoding and recognition data to a file for txt separately
save_three_column_file(Encoding_Data_Face_Block_2_Run_2, os.path.join(output_directory, f'encoding_data_face_run2_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Place_Block_2_Run_2, os.path.join(output_directory, f'encoding_data_place_run2_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Pair_Block_2_Run_2, os.path.join(output_directory, f'encoding_data_pair_run2_P{participant_number}.txt'))

save_three_column_file(Recog_Data_Face_Block_2_Run_2, os.path.join(output_directory, f'recog_data_face_block2_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Place_Block_2_Run_2, os.path.join(output_directory, f'recog_data_place_block2_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Pair_Block_2_Run_2, os.path.join(output_directory, f'recog_data_pair_block2_P{participant_number}.txt'))

# Convert the remaining encoding and recognition response data in the dictionaries to lists and save them to text files
save_three_column_file(list(Encoding_Response_Data_Face_Block_2_Run_2.values()), os.path.join(output_directory, f'encoding_response_data_face_run2_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Place_Block_2_Run_2.values()), os.path.join(output_directory, f'encoding_response_data_place_run2_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Pair_Block_2_Run_2.values()), os.path.join(output_directory, f'encoding_response_data_pair_run2_P{participant_number}.txt'))

# Save the recognition data in the lists to text files
save_three_column_file(Recog_Response_Data_Face_Block_2_Run_2, os.path.join(output_directory, f'recog_response_data_face_run2_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Place_Block_2_Run_2, os.path.join(output_directory, f'recog_response_data_place_run2_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Pair_Block_2_Run_2, os.path.join(output_directory, f'recog_response_data_pair_run2_P{participant_number}.txt'))

# BLOCK 3 RUN 3
# Save the encoding and recognition data to a file for txt separately
save_three_column_file(Encoding_Data_Face_Block_3_Run_3, os.path.join(output_directory, f'encoding_data_face_run3_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Place_Block_3_Run_3, os.path.join(output_directory, f'encoding_data_place_run3_P{participant_number}.txt'))
save_three_column_file(Encoding_Data_Pair_Block_3_Run_3, os.path.join(output_directory, f'encoding_data_pair_run3_P{participant_number}.txt'))

save_three_column_file(Recog_Data_Face_Block_3_Run_3, os.path.join(output_directory, f'recog_data_face_block3_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Place_Block_3_Run_3, os.path.join(output_directory, f'recog_data_place_block3_P{participant_number}.txt'))
save_three_column_file(Recog_Data_Pair_Block_3_Run_3, os.path.join(output_directory, f'recog_data_pair_block3_P{participant_number}.txt'))

# Convert the remaining encoding and recognition response data in the dictionaries to lists and save them to text files
save_three_column_file(list(Encoding_Response_Data_Face_Block_3_Run_3.values()), os.path.join(output_directory, f'encoding_response_data_face_run3_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Place_Block_3_Run_3.values()), os.path.join(output_directory, f'encoding_response_data_place_run3_P{participant_number}.txt'))
save_three_column_file(list(Encoding_Response_Data_Pair_Block_3_Run_3.values()), os.path.join(output_directory, f'encoding_response_data_pair_run3_P{participant_number}.txt'))

# Save the recognition data in the lists to text files
save_three_column_file(Recog_Response_Data_Face_Block_3_Run_3, os.path.join(output_directory, f'recog_response_data_face_run3_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Place_Block_3_Run_3, os.path.join(output_directory, f'recog_response_data_place_run3_P{participant_number}.txt'))
save_three_column_file(Recog_Response_Data_Pair_Block_3_Run_3, os.path.join(output_directory, f'recog_response_data_pair_run3_P{participant_number}.txt'))



# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
