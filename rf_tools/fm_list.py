import time
import random
import numpy as np
import pyvisa

# Same frequency grid and power range as fm.py, but the SMA100B steps through the
# points itself (list mode) with a hardware dwell, instead of Python stepping it.

SMA_IP_ADDRESS = "192.168.24.10"
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"

START_FREQ = 212.8e6
STOP_FREQ = 213.4e6
NUM_POINTS = 500
DWELL_S = 0.002          # hardware dwell per point, minimum about 1 ms
DURATION_S = 0.0         # 0 = run until Enter / Ctrl+C

MIN_POWER = -20.0
MAX_POWER = 0.0

LIST_NAME = "fm_sweep"

frequencies = np.linspace(START_FREQ, STOP_FREQ, NUM_POINTS)
powers = [random.uniform(MIN_POWER, MAX_POWER) for _ in frequencies]

rm = pyvisa.ResourceManager("@py")
sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)
print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")

try:
    sma.write("*RST")
    sma.write("*CLS")

    print(f"Loading {NUM_POINTS} points into list '{LIST_NAME}'...")
    setup_start = time.perf_counter()
    sma.write(f'SOURce1:LIST:SELect "{LIST_NAME}"')   # creates the list if it does not exist
    sma.write("SOURce1:LIST:FREQuency " + ",".join(f"{f:.1f}" for f in frequencies))
    sma.write("SOURce1:LIST:POWer " + ",".join(f"{p:.2f}" for p in powers))
    sma.write("SOURce1:LIST:DWELl:MODE GLOBal")        # one dwell for every point
    sma.write(f"SOURce1:LIST:DWELl {DWELL_S:.6f}")
    sma.write("SOURce1:LIST:MODE AUTO")
    sma.write("SOURce1:LIST:TRIGger:SOURce SINGle")    # one pass per trigger, used to time a pass
    sma.write("SOURce1:FREQuency:MODE LIST")

    loaded = int(sma.query("SOURce1:LIST:FREQuency:POINts?"))
    error = sma.query("SYSTem:ERRor?").strip()
    setup_time = time.perf_counter() - setup_start
    print(f"Points loaded: {loaded}/{NUM_POINTS} | Error queue: {error}")
    print(f"Setup (upload + checks) time: {setup_time:.3f} s")
    if loaded != NUM_POINTS or not error.startswith("0"):
        raise RuntimeError("Generator list setup failed, see the error queue above.")

    sma.write("OUTPut1:STATe ON")

    # Time one pass: trigger it, then poll until the generator reports the list is no longer running.
    pass_start = time.perf_counter()
    sma.write("SOURce1:LIST:TRIGger:EXECute")
    while int(float(sma.query("SOURce1:LIST:RUNNing?"))) and time.perf_counter() - pass_start < 30:
        pass
    pass_time = time.perf_counter() - pass_start
    print(f"Pass time: {pass_time:.3f} s measured (expected {NUM_POINTS * DWELL_S:.3f} s = "
          f"{NUM_POINTS} x {DWELL_S * 1e3:g} ms; includes one LAN round trip of polling)")
    print(f"Setup + first pass: {setup_time + pass_time:.3f} s")

    sma.write("SOURce1:LIST:TRIGger:SOURce AUTO")      # now free-running, restarts when finished
    print(f"List sweep running: dwell {DWELL_S * 1e3:g} ms, one pass every {NUM_POINTS * DWELL_S:.3g} s.")
    if DURATION_S:
        time.sleep(DURATION_S)
    else:
        input("Press Enter (or Ctrl+C) to stop...")
except KeyboardInterrupt:
    pass
finally:
    sma.write("SOURce1:FREQuency:MODE CW")
    sma.write("OUTPut1:STATe OFF")
    sma.close()
    rm.close()

print("\nList sweep stopped, RF output off.")
