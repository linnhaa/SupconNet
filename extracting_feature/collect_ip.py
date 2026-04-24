import os
import ipaddress
from scapy.all import PcapReader, IP, IPv6

ROOT = "data_mac/Macbook_1_50"
OUTPUT_FILE = "check_ip/dst_ip_mac.txt"

unique_ips = set()

# ==============================
# Check public IP (IPv4 + IPv6)
# ==============================
def is_valid_ip(ip):

    try:
        ip_obj = ipaddress.ip_address(ip)

        if (
            ip_obj.is_private or
            ip_obj.is_loopback or
            ip_obj.is_link_local or
            ip_obj.is_multicast
        ):
            return False

        return True

    except:
        return False


# ==============================
# Load existing IPs
# ==============================
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        unique_ips = {line.strip() for line in f}

print("Existing IPs:", len(unique_ips))


# ==============================
# Process PCAP files
# ==============================
for root, dirs, files in os.walk(ROOT):
    for file in files:

        if not file.endswith(".pcap"):
            continue

        pcap_path = os.path.join(root, file)
        print("Processing:", pcap_path)

        try:
            with PcapReader(pcap_path) as packets:
                for pkt in packets:

                    if IP in pkt:

                        src = pkt[IP].src
                        dst = pkt[IP].dst

                        if is_valid_ip(src):
                            unique_ips.add(src)

                        if is_valid_ip(dst):
                            unique_ips.add(dst)

                    elif IPv6 in pkt:

                        src = pkt[IPv6].src
                        dst = pkt[IPv6].dst

                        if is_valid_ip(src):
                            unique_ips.add(src)

                        if is_valid_ip(dst):
                            unique_ips.add(dst)

        except Exception as e:
            print("Read error:", e)


# ==============================
# Save unique IPs
# ==============================
with open(OUTPUT_FILE, "w") as f:
    for ip in sorted(unique_ips):
        f.write(ip + "\n")

print("\nTotal unique IPs:", len(unique_ips))