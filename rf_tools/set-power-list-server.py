#!/usr/bin/env python3
"""TCP server: wait for a power profile from the client and load it into the generator (SMA) list mode.
The client sends newline-terminated lines (frequency in Hz, power in dBm, dwell in s, comma separated):
    SOUR:FREQ:MODE LIST
    SOUR:LIST:FREQ 213e6,214e6,215e6
    SOUR:LIST:POW -10,-20,-15
    SOUR:LIST:DWEL 0.5,0.5,1
    EOT                       # optional; otherwise the profile applies once FREQ, POW and DWEL have all arrived
Without a DWEL line, EOT applies the profile with DEFAULT_DWELL_S for every bin.
Each profile plays once (no repeat); RF then goes off and the script waits for the next one.
Profiles that arrive while one is playing are handled after it finishes. Every profile is logged to rf_events.jsonl.
RF turns off when the client disconnects, when a profile is rejected, and on Ctrl+C.
While a profile plays, tinySA_monitor.py prints the tinySA power per frequency bin, timed from the trigger.
    python set-power-list-server.py --port 11111              # --no-tinysa to run without the tinySA
"""
import argparse
import json
import socket
import threading
from datetime import datetime

import pyvisa

SMA_IP_ADDRESS = "192.168.24.10"
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"
EVENTS = "rf_events.jsonl"
LIST_NAME = "tinysa_profile"
KEYS = {"SOUR:LIST:FREQ": "freq", "SOUR:LIST:POW": "pow", "SOUR:LIST:DWEL": "dwel"}
DEFAULT_DWELL_S = 0.2  # used for every bin when a profile ends (EOT) without a dwell line


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


def send(sma, command, after_write=None):
    """Write one SCPI command; fail with the generator's own error text if it rejects it.
    after_write runs immediately after the write, before the error check."""
    print(f"  > {command}")
    sma.write(command)
    if after_write:
        after_write()
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


def apply_profile(sma, profile, on_start=None):
    """Load the lists with RF off, then start one pass of the list. Returns the pass duration in s.
    on_start is called right after the trigger is written."""
    n = len(profile["freq"])
    if not (n == len(profile["pow"]) == len(profile["dwel"])) or n == 0:
        raise ValueError(f"list lengths differ: freq {n}, pow {len(profile['pow'])}, dwel {len(profile['dwel'])}")
    send(sma, "*CLS")
    send(sma, "OUTPut:STATe OFF")
    send(sma, "SOURce:FREQuency:MODE CW")
    send(sma, "SOURce:POWer:MODE CW")
    send(sma, f"SOURce:POWer:LEVel {profile['pow'][0]}")  # RF comes on at the first list power, not an old level
    send(sma, f'SOURce:LIST:SELect "{LIST_NAME}"')
    send(sma, "SOURce:LIST:FREQuency " + ",".join(f"{v:.0f}" for v in profile["freq"]))
    send(sma, "SOURce:LIST:POWer " + ",".join(f"{v}" for v in profile["pow"]))
    if len(set(profile["dwel"])) == 1:  # the generator's DWELl takes one value for the whole list
        send(sma, f"SOURce:LIST:DWELl {profile['dwel'][0]}")
    else:
        send(sma, "SOURce:LIST:DWELl:LIST " + ",".join(f"{v}" for v in profile["dwel"]))
    for q in ("FREQuency", "POWer"):
        if int(ask(sma, f"SOURce:LIST:{q}:POINts?")) != n:
            raise RuntimeError(f"generator holds the wrong number of {q} points")
    try:  # best effort: show what the generator really holds, before the trigger
        for q in ("FREQuency", "POWer", "DWELl"):
            print(f"    {ask(sma, f'SOURce:LIST:{q}?')}")
    except RuntimeError as e:
        print(f"  readback skipped: {e}")
        sma.write("*CLS")
    send(sma, "SOURce:LIST:MODE AUTO")
    send(sma, "SOURce:LIST:TRIGger:SOURce SINGle")  # one pass per trigger, no repeat
    send(sma, "OUTPut:STATe ON")  # the generator refuses list mode while RF is off
    send(sma, "SOURce:FREQuency:MODE LIST")
    send(sma, "SOURce:LIST:TRIGger:EXECute", after_write=on_start)
    return sum(profile["dwel"])


def read_lines(soc, stop):
    """Yield newline-terminated lines until the client closes or stop is set. The 1 s timeout lets Ctrl+C through on Windows."""
    soc.settimeout(1.0)
    buf = b""
    while not stop.is_set():
        try:
            data = soc.recv(4096)
        except socket.timeout:
            continue
        if not data:
            return
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            yield raw.decode(errors="replace")


def serve(host, port, use_tinysa=True, on_event=None, stop=None):
    """Run the server until stop is set (or Ctrl+C). on_event(kind, data) reports what is happening:
        status   text      where the server is: waiting, loading, playing, ...
        profile  dict      freq_hz, power_dbm, dwell_s of the profile about to play
        start    datetime  the generator was triggered
        done     None      the pass is over, RF is off and the last sweep has been delivered
        sweep    dict      t0, t1 (s since the trigger) and rows [(bin, freq_hz, dbm, mw)] of one tinySA sweep
    """
    stop = stop or threading.Event()
    emit = on_event or (lambda kind, data=None: None)

    rm = pyvisa.ResourceManager()
    sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)
    print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")
    monitor = None
    if use_tinysa:
        from tinySA_monitor import Monitor  # opened now, so nothing is opened while a profile plays
        monitor = Monitor(on_sweep=lambda t0, t1, rows: emit("sweep", {"t0": t0, "t1": t1, "rows": rows}),
                          verbose=on_event is None)
        print("Connected to tinySA")

    def started():  # runs right after the trigger is written
        if monitor:
            monitor.go()
        emit("start", datetime.now())

    soc_svr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc_svr.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_svr.bind((host, port))
    soc_svr.listen(1)
    soc_svr.settimeout(1.0)  # so Ctrl+C and stop are noticed while waiting for a client (Windows)
    print(f"Wait on {host}:{port} for a profile (Ctrl+C to stop)...")
    emit("status", "waiting for a client")
    try:
        while not stop.is_set():
            try:
                soc, addr = soc_svr.accept()
            except socket.timeout:
                continue
            print(f"Accept: {addr}.")
            emit("status", "client connected, waiting for a profile")
            pending = {}
            with soc:
                for raw in read_lines(soc, stop):
                    line = raw.strip()
                    if not line:
                        continue
                    if line.upper() != "EOT" and not parse(line, pending):
                        print(f"ignored: {line}")
                    if line.upper() == "EOT" and "freq" in pending and "pow" in pending and "dwel" not in pending:
                        pending["dwel"] = [DEFAULT_DWELL_S] * len(pending["freq"])
                        print(f"no dwell sent, using the default {DEFAULT_DWELL_S:g} s")
                    if complete(pending):
                        event_id = f"{datetime.now():%Y%m%d_%H%M%S}"
                        try:
                            emit("profile", {"event_id": event_id, "freq_hz": pending["freq"],
                                             "power_dbm": pending["pow"], "dwell_s": pending["dwel"]})
                            emit("status", "loading the profile")
                            if monitor:
                                monitor.arm(pending["freq"], sum(pending["dwel"]))  # parked until the trigger
                            duration = apply_profile(sma, pending, on_start=started)
                            log_event("profile", event_id, freq_hz=pending["freq"], power_dbm=pending["pow"],
                                      dwell_s=pending["dwel"])
                            print(f"-> profile {event_id}: {len(pending['freq'])} bins live for {duration:g} s")
                            emit("status", f"playing {len(pending['freq'])} bins for {duration:g} s")
                            stop.wait(duration + 0.1)
                            sma.write("OUTPut:STATe OFF")
                            if monitor:
                                if stop.is_set():
                                    monitor.cancel()
                                else:
                                    monitor.wait()  # RF is already off; let the last sweep finish
                            print("profile done, RF off, waiting for the next one")
                            emit("status", "profile done, RF off, waiting for the next profile")
                            emit("done", None)
                        except (ValueError, RuntimeError, pyvisa.VisaIOError) as e:
                            sma.write("OUTPut:STATe OFF")
                            if monitor:
                                monitor.cancel()
                            print(f"profile rejected, RF off: {e}")
                            emit("status", f"profile rejected, RF off: {e}")
                        pending = {}
                    elif line.upper() == "EOT":
                        if pending:  # an EOT after a profile that already played is just the end marker
                            print("EOT with an incomplete profile, discarded")
                        pending = {}
            sma.write("OUTPut:STATe OFF")
            print("Client closed the connection, RF off.")
            emit("status", "client closed the connection, RF off")
    except KeyboardInterrupt:
        pass
    finally:
        sma.write("OUTPut:STATe OFF")
        sma.close()
        soc_svr.close()
        log_event("off", f"{datetime.now():%Y%m%d_%H%M%S}")
        if monitor:
            monitor.close()
        print("Generator safely disconnected.")
        emit("status", "stopped, RF off")


def main():
    p = argparse.ArgumentParser(description="Receive power profiles from a client and load them into the generator.")
    p.add_argument("--host", default=socket.gethostname(), help="address to listen on (default: this machine)")
    p.add_argument("--port", type=int, default=11111, help="port to listen on (default 11111)")
    p.add_argument("--no-tinysa", action="store_true", help="do not read the tinySA while a profile plays")
    a = p.parse_args()
    serve(a.host, a.port, not a.no_tinysa)


if __name__ == "__main__":
    main()
