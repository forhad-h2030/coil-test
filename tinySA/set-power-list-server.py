#!/usr/bin/env python3
"""TCP server: wait for a power profile from the client and load it into the generator (SMA) list mode.
The client sends newline-terminated lines (frequency in Hz, power in dBm, dwell in s, comma separated):
    SOUR:FREQ:MODE LIST
    SOUR:LIST:FREQ 213e6,214e6,215e6
    SOUR:LIST:POW -10,-20,-15
    SOUR:LIST:DWEL 0.5,0.5,1
    EOT                       # optional; otherwise the profile applies once FREQ, POW and DWEL have all arrived
Each profile plays once (no repeat); RF then goes off and the script waits for the next one.
Profiles that arrive while one is playing are handled after it finishes. Every profile is logged to rf_events.jsonl.
RF turns off when the client disconnects, when a profile is rejected, and on Ctrl+C.
    python set-power-list-server.py --port 11111
"""
import argparse
import json
import socket
import time
from datetime import datetime

import pyvisa

SMA_IP_ADDRESS = "192.168.24.10"
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"
EVENTS = "rf_events.jsonl"
LIST_NAME = "tinysa_profile"
KEYS = {"SOUR:LIST:FREQ": "freq", "SOUR:LIST:POW": "pow", "SOUR:LIST:DWEL": "dwel"}


def log_event(kind, event_id, **fields):
    line = {"event_id": event_id, "kind": kind, "t": datetime.now().isoformat(), **fields}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(line) + "\n")


def parse(line, profile):
    """Fold one server line into the pending profile. Returns True when the line was understood."""
    cmd, _, arg = line.strip().partition(" ")
    cmd = cmd.upper()
    if cmd in KEYS:
        profile[KEYS[cmd]] = [float(v) for v in arg.split(",") if v.strip()]
    elif cmd == "SOUR:FREQ:MODE":
        profile["mode"] = arg.strip().upper()
    else:
        return False
    return True


def complete(profile):
    return all(k in profile for k in KEYS.values())


def send(sma, command):
    """Write one SCPI command; fail with the generator's own error text if it rejects it."""
    print(f"  > {command}")
    sma.write(command)
    err = sma.query("SYSTem:ERRor?").strip()
    if not err.startswith("0"):
        raise RuntimeError(f"{command!r} rejected: {err}")


def ask(sma, command):
    """Query the generator; name the command if it never answers."""
    print(f"  > {command}")
    try:
        return sma.query(command).strip()
    except pyvisa.VisaIOError as e:
        raise RuntimeError(f"{command!r} got no answer: {e}") from e


def apply_profile(sma, profile):
    """Load the lists with RF off, then start one pass of the list. Returns the pass duration in s."""
    n = len(profile["freq"])
    if not (n == len(profile["pow"]) == len(profile["dwel"])) or n == 0:
        raise ValueError(f"list lengths differ: freq {n}, pow {len(profile['pow'])}, dwel {len(profile['dwel'])}")
    send(sma, "*CLS")
    send(sma, "OUTPut:STATe OFF")
    send(sma, "SOURce:FREQuency:MODE CW")
    send(sma, f'SOURce:LIST:SELect "{LIST_NAME}"')
    send(sma, "SOURce:LIST:FREQuency " + ",".join(f"{v:.0f}" for v in profile["freq"]))
    send(sma, "SOURce:LIST:POWer " + ",".join(f"{v}" for v in profile["pow"]))
    send(sma, "SOURce:LIST:DWELl " + ",".join(f"{v}" for v in profile["dwel"]))
    for q in ("FREQuency", "POWer", "DWELl"):
        if int(ask(sma, f"SOURce:LIST:{q}:POINts?")) != n:
            raise RuntimeError(f"generator holds the wrong number of {q} points")
    send(sma, "SOURce:LIST:MODE AUTO")
    send(sma, "SOURce:LIST:TRIGger:SOURce SINGle")  # one pass per trigger, no repeat
    send(sma, "SOURce:FREQuency:MODE LIST")
    send(sma, "OUTPut:STATe ON")
    send(sma, "SOURce:LIST:TRIGger:EXECute")
    return sum(profile["dwel"])


def main():
    p = argparse.ArgumentParser(description="Receive power profiles from a client and load them into the generator.")
    p.add_argument("--host", default=socket.gethostname(), help="address to listen on (default: this machine)")
    p.add_argument("--port", type=int, default=11111, help="port to listen on (default 11111)")
    a = p.parse_args()

    rm = pyvisa.ResourceManager()
    sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)
    print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")
    soc_svr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc_svr.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_svr.bind((a.host, a.port))
    soc_svr.listen(1)
    print(f"Wait on {a.host}:{a.port} for a profile (Ctrl+C to stop)...")
    try:
        while True:
            soc, addr = soc_svr.accept()
            print(f"Accept: {addr}.")
            pending = {}
            with soc:
                for raw in soc.makefile("r", newline="\n"):
                    line = raw.strip()
                    if not line:
                        continue
                    if line.upper() != "EOT" and not parse(line, pending):
                        print(f"ignored: {line}")
                    if complete(pending):
                        event_id = f"{datetime.now():%Y%m%d_%H%M%S}"
                        try:
                            duration = apply_profile(sma, pending)
                            log_event("profile", event_id, freq_hz=pending["freq"], power_dbm=pending["pow"],
                                      dwell_s=pending["dwel"])
                            print(f"-> profile {event_id}: {len(pending['freq'])} bins live for {duration:g} s")
                            time.sleep(duration + 0.1)
                            sma.write("OUTPut:STATe OFF")
                            print("profile done, RF off, waiting for the next one")
                        except (ValueError, RuntimeError, pyvisa.VisaIOError) as e:
                            sma.write("OUTPut:STATe OFF")
                            print(f"profile rejected, RF off: {e}")
                        pending = {}
                    elif line.upper() == "EOT":
                        print("EOT with an incomplete profile, discarded")
                        pending = {}
            sma.write("OUTPut:STATe OFF")
            print("Client closed the connection, RF off.")
    except KeyboardInterrupt:
        pass
    finally:
        sma.write("OUTPut:STATe OFF")
        sma.close()
        soc_svr.close()
        log_event("off", f"{datetime.now():%Y%m%d_%H%M%S}")
        print("Generator safely disconnected.")


if __name__ == "__main__":
    main()
