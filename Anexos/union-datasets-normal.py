from scapy.all import rdpcap, wrpcap, Ether, IP, TCP, UDP
from scapy.config import conf

conf.max_list_count = 100000
#Carga de datasets 

pcap1 = rdpcap("normalDt/Normal-h3_2.pcap")
pcap2 = rdpcap("normalDt/normalDataset.pcap")

# Diccionario de IP y MAC

ip_mapping = { '192.168.200.4': '192.168.20.133' } #Máquina atacante

mac_mapping = { '08:00:27:39:ed:f1': '00:00:00:00:00:03'}

fusion = pcap1 + pcap2
# Manipulación de campos de dataset

for pkt in fusion:

    if IP in pkt:
        pkt[IP].tos |= 0b00000001 #Modificación del bit de ECN

        if pkt[IP].src in ip_mapping:
            pkt[IP].src = ip_mapping[pkt[IP].src]
        
        if pkt[IP].dst in ip_mapping:
            pkt[IP].dst = ip_mapping[pkt[IP].dst]
        
        del pkt[IP].chksum

        if TCP in pkt:
            del pkt[TCP].chksum 
            pkt = IP(bytes(pkt)) 
        
        elif UDP in pkt:
            del pkt[UDP].chksum
            pkt = IP(bytes(pkt))   

    
    if Ether in pkt:
        if pkt[Ether].src in mac_mapping:
            pkt[Ether].src = mac_mapping[pkt[Ether].src]
        
        if pkt[Ether].dst in mac_mapping:
            pkt[Ether].dst = mac_mapping[pkt[Ether].dst]
    
    

wrpcap('Output/normalDt-fusionado.pcap', fusion)


