#!/usr/bin/env python3
"""Hold the RF generator on one CW frequency and power until you press Enter, then switch RF off.
Each on and off is logged to rf_events.jsonl. It never touches the tinySA.
    python set_cw_tone.py --freq 213 --power -10      # frequency in MHz, power in dBm
"""
import argparse
import json
from datetime import datetime

import pyvisa

SMA_IP_ADDRESS = "192.168.24.10"
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"
EVENTS = "rf_events.jsonl"
FREQ_MHZ = 213.0
POWER_DBM = -10.0


def log_event(kind, event_id, freq_mhz, power_dbm):
    """Append one event line; the recorder acts on kind == 'on'."""
    line = {"event_id": event_id, "kind": kind, "t": datetime.now().isoformat(),
            "freq_mhz": freq_mhz, "power_dbm": power_dbm}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(line) + "\n")


p = argparse.ArgumentParser(description="Set the generator to one frequency and power.")
p.add_argument("--freq", type=float, default=FREQ_MHZ, help=f"frequency, MHz (default {FREQ_MHZ:g})")
p.add_argument("--power", type=float, default=POWER_DBM, help=f"power, dBm (default {POWER_DBM:g})")
a = p.parse_args()
target_freq_mhz = a.freq
target_power_dbm = a.power

rm = pyvisa.ResourceManager()
sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)
print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")
event_id = f"{datetime.now():%Y%m%d_%H%M%S}"

try:
    sma.write("*RST")

    target_freq_hz = target_freq_mhz * 1e6
    print(f"\nSetting single frequency bin to {target_freq_mhz} MHz at {target_power_dbm} dBm...")
    sma.write(f"SOURce:FREQuency:CW {target_freq_hz}")
    sma.write(f"SOURce:POWer:LEVel {target_power_dbm}")
    sma.write("OUTPut:STATe ON")
    sma.query("*OPC?")  # the generator has finished applying the settings
    log_event("on", event_id, target_freq_mhz, target_power_dbm)

    current_freq = float(sma.query("SOURce:FREQuency:CW?"))
    current_pow = float(sma.query("SOURce:POWer:LEVel?"))
    print("\nVerification from Hardware:")
    print(f"-> Active Frequency: {current_freq / 1e6:.2f} MHz")
    print(f"-> Active Power Level: {current_pow:.2f} dBm")
    print(f"-> Event id: {event_id}")

    input(f"\nRF Output is LIVE at {target_freq_mhz} MHz @ {target_power_dbm} dBm. "
          "Press Enter to safely turn off output and close...")
finally:
    sma.write("OUTPut:STATe OFF")
    sma.close()
    log_event("off", event_id, target_freq_mhz, target_power_dbm)
    print("Generator safely disconnected.")
