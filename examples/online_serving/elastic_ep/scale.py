#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import sys

import requests


def scale(host, port, new_dp_size, num_redundant_experts=None):
    url = f"http://{host}:{port}/scale_elastic_ep"
    payload = {"new_data_parallel_size": new_dp_size}
    if num_redundant_experts is not None:
        payload["new_num_redundant_experts"] = num_redundant_experts
    headers = {"Content-Type": "application/json"}

    print(f"Sending scale request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("Scale up/down request successful!")
            return True
        else:
            print("Scale up/down request failed!")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test scale up/down functionality")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8006, help="API server port")
    parser.add_argument(
        "--new-dp-size", type=int, default=2, help="New data parallel size"
    )
    parser.add_argument(
        "--num-redundant-experts",
        type=int,
        default=None,
        help="New number of redundant experts (None = use max capacity)",
    )

    args = parser.parse_args()

    success = scale(args.host, args.port, args.new_dp_size,
                    args.num_redundant_experts)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
