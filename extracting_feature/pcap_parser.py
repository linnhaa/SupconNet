import os
import gc
from collections import Counter, defaultdict
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

# ======================================================
# CONFIG
# ======================================================

INPUT_ROOT = "data_mac/Macbook_1_50"
OUTPUT_ROOT = "data/mac"
IP_FILTER_FILE = "check_ip/dst_ip_mac.txt"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======================================================
# LOAD IP FILTER
# ======================================================

def load_ip_filter(path):
    ip_set = set()

    with open(path, "r") as f:
        for line in f:
            ip = line.strip()
            if ip:
                ip_set.add(ip)

    return ip_set


IP_SET = load_ip_filter(IP_FILTER_FILE)
print(f"[+] Loaded {len(IP_SET)} IPs for filtering")


# ======================================================
# HELPERS
# ======================================================

def extract_label(filename):
    name = filename.replace(".pcap", "")
    parts = name.split("_")

    # nếu phần cuối là số => bỏ đi
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])

    return name


def extract_ips(pkt):

    if IP in pkt:
        return pkt[IP].src, pkt[IP].dst

    if IPv6 in pkt:
        return pkt[IPv6].src, pkt[IPv6].dst

    return None, None


# ======================================================
# READ PACKETS
# ======================================================

def parse_pcap_packets(pcap_path):

    packets = []

    try:
        with PcapReader(pcap_path) as reader:

            for pkt in reader:

                src_ip, dst_ip = extract_ips(pkt)

                if src_ip is None:
                    continue

                if TCP not in pkt and UDP not in pkt:
                    continue

                # filter IP
                if src_ip not in IP_SET and dst_ip not in IP_SET:
                    continue

                packets.append({
                    "time": float(pkt.time),
                    "src": src_ip,
                    "dst": dst_ip,
                    "size": len(pkt)
                })

    except Exception as e:
        print(f"[!] Error reading {pcap_path}: {e}")

    return packets


# ======================================================
# FIND LOCAL IP
# ======================================================

def detect_local_ip(packets):

    counter = Counter()

    for p in packets:
        counter[p["src"]] += 1
        counter[p["dst"]] += 1

    if not counter:
        return None

    return counter.most_common(1)[0][0]


# ======================================================
# WRITE OUTPUT
# ======================================================

def write_flow_file(packets, output_path):

    if len(packets) < 2:
        return False

    local_ip = detect_local_ip(packets)

    if local_ip is None:
        return False

    t0 = packets[0]["time"]

    with open(output_path, "w") as f:

        # # header
        # f.write("time direction packetsize\n")

        for p in packets:

            rel_time = p["time"] - t0

            if p["src"] == local_ip:
                direction = 1
            elif p["dst"] == local_ip:
                direction = -1
            else:
                continue

            f.write(f"{rel_time:.6f} {direction} {p['size']}\n")

    return True


# ======================================================
# GROUP FILES BY LABEL
# ======================================================

label_files = defaultdict(list)

for root, dirs, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.endswith(".pcap"):
            label = extract_label(file)
            full_path = os.path.join(root, file)
            label_files[label].append(full_path)


# ======================================================
# MAIN
# ======================================================

for label in sorted(label_files.keys()):

    output_folder = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(label_files[label])

    print(f"\nProcessing label: {label}")

    counter = 1

    for pcap_path in files:

        new_name = f"{label}_{counter:03d}.txt"
        output_path = os.path.join(output_folder, new_name)

        print(f"  [+] {os.path.basename(pcap_path)}")

        packets = parse_pcap_packets(pcap_path)

        success = write_flow_file(packets, output_path)

        if success:
            print(f"      -> saved {new_name}")
            counter += 1
        else:
            print("      -> skipped")

        gc.collect()

print("\n✅ Done.")