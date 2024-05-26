#!/bin/bash

interface="enp139s0f0"
# interface="enp44s0f0"
interval=1  # Interval in seconds

while true; do
    # Capture the initial transmitted and received bytes
    RX1=$(ip -s link show $interface | grep -A 1 "RX:" | tail -n 1 | awk '{print $1}')
    TX1=$(ip -s link show $interface | grep -A 1 "TX:" | tail -n 1 | awk '{print $1}')
    
    # Wait for the specified interval
    sleep $interval
    
    # Capture the new transmitted and received bytes
    RX2=$(ip -s link show $interface | grep -A 1 "RX:" | tail -n 1 | awk '{print $1}')
    TX2=$(ip -s link show $interface | grep -A 1 "TX:" | tail -n 1 | awk '{print $1}')
    
    # Calculate the difference and convert to Mbps
    RX_BYTES=$(($RX2 - $RX1))
    TX_BYTES=$(($TX2 - $TX1))
    RX_Mbps=$(echo "scale=2; $RX_BYTES * 8 / $interval / 1000000" | bc)
    TX_Mbps=$(echo "scale=2; $TX_BYTES * 8 / $interval / 1000000" | bc)
    
    # Display the results
    clear
    echo "Interface: $interface"
    echo "Download: $RX_Mbps Mbps"
    echo "Upload: $TX_Mbps Mbps"
    echo "------------------------"
done
