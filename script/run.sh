#!/bin/sh

python3 /Users/dhruv/Desktop/MediX/main.py &

sleep 5

open -a Firefox "http://127.0.0.1:5000"

echo "Check browser"
