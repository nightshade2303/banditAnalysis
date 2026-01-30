"""
Record audio (.wav) + video (.h264) chunks of specified duration simultaneously
and check every set amount of time for .wav and .h264
files to be transferred to the NAS.
.h264 files are converted to .mp4 files before the transfer.
"""

# 2024-08-02 Rishika's version for ephys - make 40 min recordings for the session and keep transferring them to server
#NOTE: CURRENT VERSION OF THIS SCRIPT TRANSFERS IN BETWEEN, CHANGE THIS BEFORE DEPLOYING

import concurrent.futures
from datetime import datetime
import logging
import ntplib
import os
# import pyaudio
import re
import subprocess
import time
import RPi.GPIO as GPIO
# import wave

CAMERABCM = 26
RECORD_FOR = 0.4  # (hours)
CHUNK_SIZE = 0.5  # (minutes) the length of each audio/video clip
FRAMES = 2700 # same as above but number of frames
TRANSFER_PERIOD = 2  # (minutes) sets convert and upload frequency
WORKING_DIR = "/home/pi/AutoTrainerModular/"  # directory in which all the files are
NAS_DIR = "/home/pi/data/Video/"  # dir where the videos are stored (in the NAS)
NAS_MOUNT_DIR = "/home/pi/data/"  # dir where the NAS is mounted
USERNAME = "rishika.sharma/4portProb/Boxephys/Shinx/"  # NAS username

def setupGPIO():
    # setting up a GPIO pin on the Pi
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(CAMERABCM, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(CAMERABCM, GPIO.RISING, interruptGPIO, bouncetime = 100) # detect rising gpio, bounce tau = 100ms 

def interruptGPIO():
    # read the last number written to the filename.frames, copy it and write it along with the current timestamp to filename.events
    filename = str(datetime.now().date()) + "__" + str(datetime.now().time())[:-7] + ".frames"
    filename = filename.replace(":", "_").replace("-", "_")
    frametime = (subprocess.check_output(['tail', '-1', filename])).decode('utf-8').strip('\n')
    logging.debug(f"TTL:     event detected + {frametime}")

def createNewLogfile():
    oldLogs = re.findall("playRecord%d+.log", " ".join(os.listdir()))
    logging.basicConfig(filename="playRecord%d.log" % (len(oldLogs)), format="%(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def recordVideo(seconds):

    filename = str(datetime.now().date()) + "__" + str(datetime.now().time())[:-7] + ".h264"
    filename = filename.replace(":", "_").replace("-", "_")

    #how to insert a custom output file name in the command below (replace test1332.h264 with the filename)
    
    logging.debug("VIDEO:     started recording")
    recordVideo = f"rpivid --level 4.2 --width 1332 --height 990 --mode 1332:990 --framerate 90 --frames {FRAMES} -o {filename} -n --save-pts {filename.strip('.h264')+'.frames'} --denoise cdn_off"
    
    #print(recordVideo)
    # raspivid -t 0 -o - | tee /video.h264  | nc 192.168.178.30 5001 - link https://forums.raspberrypi.com/viewtopic.php?t=72405

    #os.system(recordVideo)
    subprocess.run(recordVideo, shell=True)
    logging.debug("VIDEO:     stopped recording")

# def recordAudio(seconds, audioMothIndex):
#     changeDirCreate(WORKING_DIR + "Created/")
#     audio_format = pyaudio.paInt16
#     number_of_channels = 1
#     sample_rate = 192000
#     chunk_size = 4096

#     audio = pyaudio.PyAudio()

#     stream = audio.open(format = audio_format, rate = sample_rate, channels = number_of_channels, \
#                         input_device_index = audioMothIndex, input = True)

#     frames_per_buffer = chunk_size

#     filename = str(datetime.now().date()) + "__" + str(datetime.now().time())[:-7] + ".wav"
#     filename = filename.replace(":", "_").replace("-", "_")
#     logging.debug(f"AUDIO:     {filename} started recording")

#     data = []
#     total_samples = sample_rate * seconds

#     while total_samples > 0:
#         samples = min(total_samples, chunk_size)
#         data.append(stream.read(samples, exception_on_overflow=False))
#         total_samples -= samples


#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     wavefile = wave.open(filename, "wb")
#     wavefile.setnchannels(number_of_channels)
#     wavefile.setsampwidth(audio.get_sample_size(audio_format))
#     wavefile.setframerate(sample_rate)
#     wavefile.writeframes(b"".join(data))
#     wavefile.close()

# def findAudioMoth():
#     deviceIndex = None
#     audio = pyaudio.PyAudio()

#     for i in range(audio.get_device_count()):
#         if "AudioMoth" in audio.get_device_info_by_index(i).get("name"):
#             logging.debug("AUDIO:     AudioMoth found")
#             deviceIndex = i
#             break

#     if deviceIndex == None:
#         logging.debug('AUDIO:     no AudioMoth found')

#     return deviceIndex

def convertAndTransfer(skipLast, skipFirst):
    changeDirCreate(WORKING_DIR + "Created/")
    files = " ".join(os.listdir())
    videoFiles = re.findall(r".[^ ]*.h264", files)
    videoFiles = [vid.strip() for vid in videoFiles]
    videoFiles.sort()
    audioFiles = re.findall(r".[^ ]*.wav", files)
    audioFiles = [audio.strip() for audio in audioFiles]
    audioFiles.sort()

    if not skipFirst:
        cmd = "sudo mv *.mp4 %s" %(NAS_DIR)
        try:
            subprocess.run(cmd.split())
            logging.debug("TRANSFER:  transferred existing mp4 & wav files")
        except Exception as e:
            logging.debug("TRANSFER:  failed to transfer existing mp4 and wav files")
            logging.debug(e)

    if len(videoFiles) == 0 and len(audioFiles) == 0:
        logging.debug("TRANSFER:  No A/V files found this time, skipping")
    else:
        logging.debug("TRANSFER:  (%s) video files to be transferred" %(len(videoFiles)))
        for video in (videoFiles[:-1] if skipLast else videoFiles):
            # videoFiles[:-1] -> don't transfer the last video/audio since it might still be incomplete
            video = video[:-5]

            cmd = "ffmpeg -framerate 90 -i %s.h264 -c copy %s.mp4 -hide_banner -loglevel error" %(video, video) # convert vid to mp4
            try:
                subprocess.run(cmd.split())
                logging.debug("TRANSFER:  converted %s.h264 to mp4" %video)
            except Exception as e:
                logging.debug("TRANSFER:  failed to convert %s.h264 to mp4" %video)
                logging.debug(e)

            cmd = "rm %s.h264" %video  # remove old h264 vid
            try:
                subprocess.run(cmd.split())
                logging.debug("TRANSFER:  removed old %s.h264" %video)
            except Exception as e:
                logging.debug("TRANSFER:  failed to remove old %s.h264" %video)
                logging.debug(e)

            cmd = "sudo mv %s.mp4 %s" %(video, NAS_DIR)  # move new mp4 to NAS
            try:
                subprocess.run(cmd.split())
                logging.debug("TRANSFER:  moved %s.mp4 to %s" %(video, NAS_DIR))
            except Exception as e:
                logging.debug("TRANSFER:  failed to move %s.mp4 to %s" %(video, NAS_DIR))
                logging.debug(e)

        # logging.debug(f"TRANSFER:  ({len(audioFiles) - 1}) audio files to be transferred")
        # for audio in (audioFiles[:-1] if skipLast else audioFiles):
        #     cmd = f"sudo mv {audio} {NAS_DIR}"  # audio files require no conversion
        #     try:
        #         subprocess.run(cmd.split())
        #         logging.debug(f"TRANSFER:  moved {audio} to {NAS_DIR}")
        #     except Exception as e:
        #         logging.debug("TRANSFER:  failed to move {audio} to {NAS_DIR}")
    if not skipLast:
        logging.debug("TRANSFER:  final transfer complete")

def updateRpiTime():
    client = ntplib.NTPClient()
    response = client.request("ntp.iisc.ac.in")
    timeString = time.ctime(response.tx_time)
    cmd = "sudo date -s %s" %(timeString)

    try:
        subprocess.run(cmd.split())
        logging.debug("TIME:      RPi time updated to %s" %(timeString))
    except Exception as e:
        logging.debug("TIME:      could not update rpi time")
        logging.debug("TIME:      command raised an error")
        logging.debug("TIME:      %s"%(e))

def mountNas(username, nasIp, credentials="/home/pi/.smbcredentials"):
    changeDirCreate(NAS_MOUNT_DIR)
    files = os.listdir()

    if "#recycle" in files:
        logging.debug("NAS:       already mounted")
        return

    logging.debug("NAS:       not mounted, mounting now")
    cmd = "sudo mount -t cifs //%s/%s %s -o credentials=%s" %(nasIp, username, NAS_MOUNT_DIR, credentials)

    try:
        subprocess.run(cmd.split())
    except Exception as e:
        logging.debug("NAS:       mounting raised an error")
        logging.debug("NAS:       %s" %e)

def recordAVandTransfer(duration, startTime):
    chunkSizeSeconds = int(CHUNK_SIZE * 60)
    transferPeriodSeconds = int(TRANSFER_PERIOD * 60)
    mothDeviceIndex = None #mothDeviceIndex = findAudioMoth()
    shortFutures = []
    longFutures = []

    # start audio+video record and convert+transfer processes in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        avCounter = 0
        transferCounter = 0 
        while time.time() - startTime < (duration * 60 * 60):
            if (time.time() - startTime) // chunkSizeSeconds >= avCounter:
                avCounter += 1
                shortFutures.append(executor.submit(recordVideo, chunkSizeSeconds))
                #shortFutures.append(executor.submit(recordAudio, chunkSizeSeconds, mothDeviceIndex)) # audio recording commented
                # add GPIO acquisition here?
                #shortFutures.append(executor.submit(GPIOPIN))

            if (time.time() - startTime) // transferPeriodSeconds >= transferCounter:
                transferCounter += 1
                longFutures.append(executor.submit(convertAndTransfer, True, True))

            for fut in concurrent.futures.as_completed(shortFutures):
                fut.result()

            #take this out of the while loop to transfer files at the very end of recording period
            for fut in concurrent.futures.as_completed(longFutures):
                fut.result()

        logging.debug("AV:        %s audio & video clips recorded" %(avCounter))

def changeDirCreate(path):
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.makedirs(path)
        os.chdir(path)

def main():
    # do not overwrite logfiles from previous runs, gives an idea of number of reboots
    createNewLogfile()
    logging.debug("### START ###")

    # change working directory to where av files are recorded to before their transfer
    changeDirCreate(WORKING_DIR)

    # update rpi time using NTP
    updateRpiTime()

    # setup GPIO
    setupGPIO()

    # check if the NAS is mounted, if not do so
    mountNas(USERNAME, "10.36.22.120")

    # transfer pre-existing .mp4 & .wav files
    convertAndTransfer(skipLast=True, skipFirst=False)

    # start recording and transfer created files to the NAS, both procs run in parallel
    recordAVandTransfer(RECORD_FOR, time.time())

    # convert and transfer the last remaining files
    convertAndTransfer(skipLast=False, skipFirst=False)

    logging.debug("### END ###")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.debug(e)
