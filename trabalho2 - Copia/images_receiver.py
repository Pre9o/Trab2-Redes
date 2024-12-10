import scapy.all as scapy
import os
import threading

cont = 0

def receive_image():
    def handle_packet(packet):
        global cont
        if packet.haslayer(scapy.UDP) and packet[scapy.UDP].dport == 12345:
            if packet.haslayer(scapy.Raw):
                data = packet[scapy.Raw].load
                print(f"Recebido pacote de {packet[scapy.IP].src} com {len(data)} bytes")
                if data == b'EOF':
                    print("Imagem recebida com sucesso.")
                    cont += 1
                else:
                    with open(f'received_images/received_image_{cont}.jpg', 'ab') as f:
                        f.write(data)
                    print("Pacote recebido e escrito no arquivo.")

    print("Aguardando pacotes...")
    scapy.sniff(iface='h2-eth0', prn=handle_packet)

def start_sniffer():
    sniffer_thread = threading.Thread(target=receive_image)
    sniffer_thread.daemon = True  # Permite que a thread termine quando o programa principal terminar
    sniffer_thread.start()
    return sniffer_thread

def send_classes_to_router(router_ip, classes):
    message = ','.join(classes).encode()
    udp_packet = scapy.IP(dst=router_ip)/scapy.UDP(sport=12347, dport=12346)/scapy.Raw(load=message)
    scapy.send(udp_packet)
    print(f"Lista de classes enviada para o roteador: {classes}")

if __name__ == "__main__":
    # Remover as imagens recebidas anteriormente
    os.system('rm -rf received_images/*received*')

    sniffer_thread = start_sniffer()

    router_ip = '10.1.1.254'
    while True:
        classes = input("Digite as classes que deseja bloquear (separadas por vírgula): ").split(',')
        send_classes_to_router(router_ip, classes)