#!/usr/bin/env python3
"""FROM NMR (the model side): sends a power profile (several frequency bins) to to_gen_server.py.
The profiles are fixed lists in PROFILES below, standing in for RF_Model.py (Infer()) that will send them later.
    python from_nmr_client.py                # send the default profile 'span'
    python from_nmr_client.py one three      # send profile 'one', wait, then 'three'
"""
import socket
import sys
import time

PROFILES = {
    "one": ["SOUR:FREQ:MODE LIST", "SOUR:LIST:FREQ 213e6", "SOUR:LIST:POW -40", "SOUR:LIST:DWEL 1", "EOT"],
    "three": ["SOUR:FREQ:MODE LIST", "SOUR:LIST:FREQ 213e6,214e6,215e6",
              "SOUR:LIST:POW -40,-45,-42", "SOUR:LIST:DWEL 0.5,0.5,1", "EOT"],
    "span": ["SOUR:FREQ:MODE LIST", "SOUR:LIST:FREQ 212.8e6,212.9e6,213e6,213.1e6,213.2e6",
             "SOUR:LIST:POW -40,-45,-50,-45,-40", "SOUR:LIST:DWEL 0.5,0.5,0.5,0.5,0.5", "EOT"],
    "bad": ["SOUR:LIST:FREQ 213e6,214e6", "SOUR:LIST:POW -40", "SOUR:LIST:DWEL 1,1,1", "EOT"],
}
WAIT_S = 10  # time to watch the generator between profiles


def main():
    host_svr = socket.gethostname()
    port_svr = 11111
    names = sys.argv[1:] or ["span"]

    try:
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((host_svr, port_svr))
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        for name in names:
            for line in PROFILES[name]:
                soc.sendall((line + "\n").encode())
                print(f"sent: {line}")
            print(f"---- profile '{name}' sent, waiting {WAIT_S} s ----")
            time.sleep(WAIT_S)

        soc.close()

    except ConnectionRefusedError:
        print("Connection refused.")


if __name__ == "__main__":
    main()
