from pvrecorder import PvRecorder
for i, name in enumerate(PvRecorder.get_available_devices()):
    print(i, name)