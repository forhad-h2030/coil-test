import time
import random
import numpy as np
import pyvisa
from pyvisa_py.serial import SerialSession
from pyvisa_py.usb import USBInstrSession, USBRawSession

# from libreVNA import libreVNA

rm = pyvisa.ResourceManager()

print("USB INSTR:", USBInstrSession.list_resources())
print("USB RAW:  ", USBRawSession.list_resources())
print("SERIAL:   ", SerialSession.list_resources())
SMA_IP_ADDRESS = "192.168.1.10"   
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"

START_FREQ = 212.8e6
STOP_FREQ = 213.4e6
NUM_POINTS = 500
SETTLING_TIME = 0.015

MIN_POWER = -20.0
MAX_POWER = 0.0

frequencies = np.linspace(START_FREQ, STOP_FREQ, NUM_POINTS)

rm = pyvisa.ResourceManager()
#sma = rm.open_resource(SMA_USB_ADDRESS, timeout=5000)
sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)

print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")

#vna = libreVNA('localhost', 19542)
#:print(f"Connected to LibreVNA Application: {vna.query('*IDN?').strip()}")

#vna.cmd(":DEV:REFCLK EXT")

sma.write("*RST")
sma.write("SOURce:POWer:LEVel 1")
sma.write("OUTPut:STATe ON")

results = []
print("\nStarting synchronized sweep...")

for idx, freq in enumerate(frequencies):
    sma.write(f"SOURce:FREQuency:CW {freq}")
    time.sleep(SETTLING_TIME)
    random_power = random.uniform(MIN_POWER, MAX_POWER)
    sma.write(f"SOURce:POWer:LEVel {random_power:.2f}")

    #raw_data = vna.query(f":VNA:TRACE:DATA? S11")
    current_pow = float(sma.query("SOURce:POWer:LEVel?"))
    random_power = random.uniform(MIN_POWER, MAX_POWER)
    results.append({
        'frequency': freq,
        'power_dbm': current_pow
    #    'data': raw_data
    })
    if idx % 10 == 0 or idx == NUM_POINTS - 1:
        #print(f"Progress: Point {idx+1}/{NUM_POINTS} ({freq/1e6:.2f} MHz)")
        print(f"Freq: {freq/1e6:8.2f} MHz | Power: {current_pow:6.2f} dBm | Point {idx+1}/{NUM_POINTS}")

sma.write("OUTPut:STATe OFF")
sma.close()
print("\nSweep Complete! Generator and VNA successfully synced via local SCPI.")
