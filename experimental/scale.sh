#!/bin/bash
HOST="localhost"
PORT=8006

python3 examples/online_serving/elastic_ep/scale.py --host $HOST --port $PORT --new-dp-size 8
