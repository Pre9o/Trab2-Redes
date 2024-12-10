import scapy.all as scapy
import os
import sys
from tkinter import Tk, filedialog

host_to_ip = {'h2': '10.2.2.1'}

def send_image(image_path, server_ip):
    with open(image_path, 'rb') as f:
        data = f.read()
    
    packets = [data[i:i+1024] for i in range(0, len(data), 1024)]
    
    for packet in packets:
        udp_packet = scapy.IP(dst=server_ip)/scapy.UDP(sport=12345, dport=12345)/scapy.Raw(load=packet)
        scapy.send(udp_packet)
    
    eof_packet = scapy.IP(dst=server_ip)/scapy.UDP(sport=12345, dport=12345)/scapy.Raw(load=b'EOF')
    scapy.send(eof_packet)
    
    print("Imagem enviada com sucesso.")

def main(server_ip):
    root = Tk()
    root.withdraw() 
    
    while True:
        command = input("Digite 'send' para enviar uma imagem ou 'exit' para sair: ").strip().lower()
        if command == 'send':
            file_paths = filedialog.askopenfilenames(title="Selecione as imagens", filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
            for image_path in file_paths:
                print(f"Enviando imagem {os.path.basename(image_path)} para o servidor...")
                send_image(image_path, server_ip)
                print(f"Imagem {os.path.basename(image_path)} enviada com sucesso.")
                print()
        elif command == 'exit':
            break
        else:
            print("Comando não reconhecido. Tente novamente.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 images_sender.py <host_name>")
        sys.exit(1)

    host_name = sys.argv[1]

    if host_name not in host_to_ip:
        print("Host name not found.")
        sys.exit(1)

    server_ip = host_to_ip[host_name]

    main(server_ip)