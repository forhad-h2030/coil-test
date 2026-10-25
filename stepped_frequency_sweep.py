#!/usr/bin/env python3
"""Step an R&S SMA carrier across frequency with a selectable power profile."""
import argparse
import csv
import random
import time

import numpy as np
import pyvisa


DEFAULT_IP = "192.168.1.10"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-address", default=f"TCPIP0::{DEFAULT_IP}::inst0::INSTR")
    parser.add_argument("--start-hz", type=float, default=212.8e6)
    parser.add_argument("--stop-hz", type=float, default=213.4e6)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--settle-s", type=float, default=0.015)
    parser.add_argument("--power-mode", choices=("fixed", "linear", "random"), default="fixed",
                        help="fixed power, linear power ramp across frequency, or random test values")
    parser.add_argument("--power-dbm", type=float, default=-10.0,
                        help="fixed power in dBm (fixed mode)")
    parser.add_argument("--power-start-dbm", type=float, default=-20.0,
                        help="starting/minimum power in dBm (linear/random modes)")
    parser.add_argument("--power-stop-dbm", type=float, default=0.0,
                        help="ending/maximum power in dBm (linear/random modes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="optional random seed for repeatable random-mode runs")
    parser.add_argument("--csv", default="stepped_frequency_sweep.csv",
                        help="output CSV path; use an empty string to disable saving")
    args = parser.parse_args()
    if args.start_hz >= args.stop_hz:
        parser.error("--start-hz must be below --stop-hz")
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.settle_s < 0:
        parser.error("--settle-s cannot be negative")
    return args


def power_at(index, count, args, rng):
    if args.power_mode == "fixed":
        return args.power_dbm
    if args.power_mode == "linear":
        fraction = index / (count - 1)
        return args.power_start_dbm + fraction * (args.power_stop_dbm - args.power_start_dbm)
    return rng.uniform(min(args.power_start_dbm, args.power_stop_dbm),
                       max(args.power_start_dbm, args.power_stop_dbm))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    frequencies = np.linspace(args.start_hz, args.stop_hz, args.points)
    resource_manager = pyvisa.ResourceManager()
    generator = None
    try:
        generator = resource_manager.open_resource(args.visa_address, timeout=5000)
        print(f"Connected to RF generator: {generator.query('*IDN?').strip()}")
        generator.write("*RST")
        generator.write("*CLS")
        generator.write("SOURce:FREQuency:CW {:.1f}".format(frequencies[0]))
        generator.write("OUTPut:STATe ON")
        rows = []
        print(f"Stepping {args.points} points; power mode: {args.power_mode}")
        for index, frequency_hz in enumerate(frequencies):
            power_dbm = power_at(index, len(frequencies), args, rng)
            generator.write(f"SOURce:FREQuency:CW {frequency_hz:.1f}")
            generator.write(f"SOURce:POWer:LEVel {power_dbm:.2f}")
            time.sleep(args.settle_s)
            actual_power_dbm = float(generator.query("SOURce:POWer:LEVel?"))
            rows.append((index, frequency_hz, power_dbm, actual_power_dbm))
            if index % 10 == 0 or index == len(frequencies) - 1:
                print(f"{index + 1:4d}/{len(frequencies)}  {frequency_hz / 1e6:10.6f} MHz  "
                      f"set {power_dbm:7.2f} dBm  readback {actual_power_dbm:7.2f} dBm")

        if args.csv:
            with open(args.csv, "w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["point", "frequency_hz", "power_set_dbm", "power_readback_dbm",
                                 "power_mode", "settle_s"])
                writer.writerows((*row, args.power_mode, args.settle_s) for row in rows)
            print(f"Saved {args.csv}")
    finally:
        if generator is not None:
            try:
                generator.write("OUTPut:STATe OFF")
            finally:
                generator.close()
        resource_manager.close()


if __name__ == "__main__":
    main()
