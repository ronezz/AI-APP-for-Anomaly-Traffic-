from scapy.all import rdpcap, wrpcap


pcap1 = rdpcap("normalDt/normalDt.pcap")
pcap2 = rdpcap("anomalyDt/anomalyDt.pcap")

fusion = pcap1 + pcap2

wrpcap('dt_fusion.pcap', fusion)
