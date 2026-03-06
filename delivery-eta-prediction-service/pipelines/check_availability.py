import argparse
import requests
import sys


def check_file_availability(month_str: str) -> bool:
    # HEAD request avoids downloading the full file just to check existence
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month_str}.parquet"

    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            print("File is available!")
            return True
        else:
            print(f"File not available. Status Code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, required=True, help="YYYY-MM")
    args = parser.parse_args()

    is_available = check_file_availability(args.month)

    # Airflow reads exit code: 0 = file ready, 1 = file not available
    if is_available:
        sys.exit(0)
    else:
        sys.exit(1)