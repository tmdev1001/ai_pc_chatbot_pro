import psutil, time, json
from datetime import datetime

usage_log = {}

def track_usage(interval=5):
    while True:
        for proc in psutil.process_iter(['pid','name']):
            name = proc.info['name']
            usage_log[name] = usage_log.get(name, 0) + interval
        with open('usage_data.json','w') as f:
            json.dump(usage_log, f)
        time.sleep(interval)

if __name__ == "__main__":
    track_usage()