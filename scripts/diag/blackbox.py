"""Black-box logger for diagnosing hard resets: one fsync'd CSV line every ~0.5 s, so the
last second before a crash survives on disk.

  python -m scripts.diag.blackbox            # runs until killed; one file per boot
  python -m scripts.diag.blackbox --report   # after a crash: summarise the last seconds
                                             # of every previous boot's log

Logged: CPU package/max-core temp, average CPU clock and busy fraction, board (acpitz)
temp, fan RPMs, adapter online + battery status/power (discharging while plugged in =
the adapter cannot keep up), GPU temp/power/SM clock/utilisation and its throttle
reasons (hw slowdown, hw thermal, hw power brake = the laptop signalling a power
shortfall, sw thermal, sw power cap). CPU package power (RAPL) needs root: not logged.
"""
import argparse, glob, os, subprocess, sys, time

OUT = "checkpoints/diag"
GPU_Q = ("temperature.gpu,power.draw,clocks.sm,utilization.gpu,"
         "clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,"
         "clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
         "clocks_event_reasons.sw_power_cap")
COLS = ("time,cpu_pkg_C,cpu_max_core_C,cpu_mhz,cpu_busy,board_C,fan_cpu,fan_gpu,fan_mid,"
        "adapter,bat_status,bat_W,gpu_C,gpu_W,gpu_mhz,gpu_util,"
        "thr_hw,thr_hw_thermal,thr_power_brake,thr_sw_thermal,thr_sw_powercap")


def read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def hwmon(name):
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if read(d + "/name").split("_")[0] == name:
            return d
    return None


def boot_id():
    return read("/proc/sys/kernel/random/boot_id")[:8]


def cpu_times():
    f = read("/proc/stat").splitlines()[0].split()[1:]
    v = list(map(int, f))
    idle = v[3] + v[4]
    return sum(v), idle


def run():
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/blackbox_{time.strftime('%Y%m%d_%H%M%S')}_{boot_id()}.csv"
    core, fans, acpi = hwmon("coretemp"), hwmon("asus"), hwmon("acpitz")
    core_temps = sorted(glob.glob(core + "/temp*_input")) if core else []
    freqs = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")
    gpu = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=subprocess.PIPE, text=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.write(fd, (COLS + "\n").encode()); os.fsync(fd)
    prev = cpu_times()
    for line in gpu.stdout:                                  # paced by nvidia-smi (~0.5 s)
        g = [x.strip() for x in line.split(",")]
        temps = [int(read(p, "0")) / 1000 for p in core_temps]
        pkg = temps[0] if temps else ""
        mx = max(temps[1:]) if len(temps) > 1 else ""
        mhz = sum(int(read(p, "0")) for p in freqs) / max(len(freqs), 1) / 1000
        cur = cpu_times()
        busy = 1 - (cur[1] - prev[1]) / max(cur[0] - prev[0], 1); prev = cur
        bat_w = int(read("/sys/class/power_supply/BAT0/power_now", "0")) / 1e6
        row = [time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}", pkg, mx, f"{mhz:.0f}",
               f"{busy:.2f}", int(read(acpi + "/temp1_input", "0")) / 1000 if acpi else "",
               *(read(f"{fans}/fan{i}_input") if fans else "" for i in (1, 2, 3)),
               read("/sys/class/power_supply/ADP0/online"),
               read("/sys/class/power_supply/BAT0/status"), f"{bat_w:.1f}",
               *g]
        os.write(fd, (",".join(map(str, row)) + "\n").encode())
        os.fsync(fd)


def report(n=12):
    files = sorted(glob.glob(f"{OUT}/blackbox_*.csv"), key=os.path.getmtime)
    cur = boot_id()
    for f in files:
        if f.endswith(f"_{cur}.csv"):
            continue
        lines = open(f).read().splitlines()
        print(f"== {f}  ({len(lines) - 1} samples; last {n} before the log stopped)")
        print(lines[0])
        for l in lines[-n:]:
            print(l)
        rows = [l.split(",") for l in lines[1:]]
        if rows:
            hdr = lines[0].split(",")
            def mx(c):
                i = hdr.index(c); v = [float(r[i]) for r in rows if len(r) > i and r[i] not in ("", "[N/A]")]
                return max(v) if v else None
            print("   max over the whole log: " + ", ".join(
                f"{c} {mx(c)}" for c in ("cpu_pkg_C", "cpu_max_core_C", "gpu_C", "gpu_W", "bat_W")))
            i = hdr.index("thr_power_brake")
            pb = sum(1 for r in rows if len(r) > i and r[i] == "Active")
            dis = sum(1 for r in rows if len(r) > 10 and r[10] == "Discharging" and r[9] == "1")
            print(f"   samples with GPU power brake: {pb}; battery discharging while plugged in: {dis}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    report() if a.report else run()
