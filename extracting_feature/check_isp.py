import requests
import time
import random

INPUT_FILE = "check_ip/dst_ip_mac.txt"
OUTPUT_FILE = "check_ip/dst_ip_with_isp.txt"

def get_isp(ip):
    url = f"http://ip-api.com/json/{ip}?fields=status,country,isp,org,as,query"
    try:
        res = requests.get(url, timeout=5)

        # check response
        if res.status_code != 200 or not res.text.strip():
            return {"ip": ip, "isp": "EMPTY"}

        data = res.json()

        if data.get("status") == "success":
            return {
                "ip": data.get("query"),
                "isp": data.get("isp", ""),
                "org": data.get("org", ""),
                "as": data.get("as", ""),
                "country": data.get("country", "")
            }
        else:
            return {"ip": ip, "isp": "FAILED"}

    except Exception as e:
        print(f"Error checking {ip}: {e}")
        return {"ip": ip, "isp": "ERROR"}

# Đọc IP
with open(INPUT_FILE, "r") as f:
    ips = [line.strip() for line in f if line.strip()]

results = []

print(f"Đang kiểm tra {len(ips)} IP...")

for i, ip in enumerate(ips):
    
    # retry 3 lần nếu fail
    for attempt in range(3):
        info = get_isp(ip)
        if info["isp"] not in ["ERROR", "EMPTY"]:
            break
        time.sleep(2)

    results.append(info)
    print(f"[{i+1}/{len(ips)}] {ip} → {info['isp']}")

    # 🔥 delay ngẫu nhiên (quan trọng nhất)
    time.sleep(random.uniform(1.2, 2.0))

# Ghi file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        line = f"{r['ip']} | ISP: {r.get('isp','')} | ORG: {r.get('org','')} | AS: {r.get('as','')} | Country: {r.get('country','')}\n"
        f.write(line)

print(f"\nĐã lưu tại: {OUTPUT_FILE}")