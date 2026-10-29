#!/usr/bin/env python3
"""Configure R&S SMA internal FM with a fixed RF power level."""
import argparse

import pyvisa


DEFAULT_IP = "192.168.24.10"
WAVEFORMS = ("SINE", "SQUare", "PULSe", "TRIangle", "TRAPeze")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-address", default=f"TCPIP0::{DEFAULT_IP}::inst0::INSTR")
    parser.add_argument("--carrier-hz", type=float, default=213e6)
    parser.add_argument("--power-dbm", type=float, default=-10.0,
                        help="fixed RF output power in dBm")
    parser.add_argument("--fm-span-hz", type=float, default=400e3,
                        help="peak-to-peak frequency excursion in Hz")
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--waveform", choices=WAVEFORMS, default="TRIangle")
    parser.add_argument("--fm-mode", choices=("HBANdwidth", "LNOise"), default="HBANdwidth")
    parser.add_argument("--off", action="store_true", help="turn FM and RF output off, then exit")
    return parser.parse_args()


def is_on(reply):
    return reply.strip().upper() in ("1", "ON")


def main():
    args = parse_args()
    resource_manager = pyvisa.ResourceManager("@py")
    generator = None
    try:
        generator = resource_manager.open_resource(args.visa_address, timeout=5000)
        print(f"Connected to RF generator: {generator.query('*IDN?').strip()}")
        if args.off:
            generator.write("SOURce1:FM1:STATe 0")
            generator.write("SOURce1:LFOutput1:STATe 0")
            generator.write("OUTPut1:STATe 0")
            print("FM, LF output, and RF output turned off.")
            return

        generator.write("*RST")
        generator.write("*CLS")
        generator.write(f"SOURce1:FREQuency:CW {args.carrier_hz:.1f}")
        generator.write(f"SOURce1:POWer:LEVel:IMMediate:AMPLitude {args.power_dbm:.3f}")
        generator.write(f"SOURce1:LFOutput1:SHAPe {args.waveform}")
        generator.write(f"SOURce1:LFOutput1:FREQuency {args.rate_hz:.6g}")
        generator.write("SOURce1:FM1:SOURce LF1")
        generator.write(f"SOURce1:FM1:DEViation {args.fm_span_hz / 2:.6g}")
        generator.write(f"SOURce1:FM:MODe {args.fm_mode}")
        generator.write("SOURce1:FM1:STATe 1")
        generator.write("SOURce1:LFOutput1:STATe 1")
        generator.write("OUTPut1:STATe 1")

        fm_state = generator.query("SOURce1:FM1:STATe?")
        lf_state = generator.query("SOURce1:LFOutput1:STATe?")
        rf_state = generator.query("OUTPut1:STATe?")
        if not (is_on(fm_state) and is_on(lf_state) and is_on(rf_state)):
            raise RuntimeError(f"Generator state check failed: FM={fm_state.strip()}, "
                               f"LF={lf_state.strip()}, RF={rf_state.strip()}")
        print(f"FM enabled: {args.waveform} at {args.rate_hz:g} Hz, "
              f"±{args.fm_span_hz / 2:g} Hz around {args.carrier_hz / 1e6:.6f} MHz")
        print(f"Fixed RF power: {args.power_dbm:.3f} dBm; RF output is ON.")
        print("The generator remains on after this script exits; rerun with --off to switch it off.")
    finally:
        if generator is not None:
            generator.close()
        resource_manager.close()


if __name__ == "__main__":
    main()
