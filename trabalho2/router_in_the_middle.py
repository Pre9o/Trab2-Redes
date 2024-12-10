import scapy.all as scapy
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
import threading

cont = 0
blocked_classes = []

def load_and_preprocess_image(image_path, img_height=128, img_width=128):
    img = Image.open(image_path).convert('RGB')  
    img = img.resize((img_height, img_width))
    # Normaliza a imagem
    img = np.array(img) / 255.0
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(image_path, classes):
    blank_image_path = "blank_images/nothing.jpg"

    original_classes = ['Athletico', 'Atletico_Goianiense', 'Atletico_Mineiro', 'Bahia', 'Botafogo', 'Bragantino', 'Corinthians', 'Criciuma', 'Cruzeiro', 'Cuiaba',
                        'Flamengo', 'Fluminense', 'Fortaleza', 'Gremio', 'Internacional', 'Juventude', 'Palmeiras', 'Sao_Paulo', 'Vasco_da_Gama', 'Vitoria']

    img_array = load_and_preprocess_image(image_path)
    model = tf.keras.models.load_model('logo_classifier2.keras')

    # Fazer a predição
    prediction = model.predict(img_array)

    print(f"Classes bloqueadas: {classes}")

    if original_classes[np.argmax(prediction)] in classes:
        return blank_image_path
    
    return image_path   


def handle_packet(pkt, classes):
    global cont

    if pkt.haslayer(scapy.UDP) and pkt[scapy.UDP].dport == 12345:
        destination_ip = pkt[scapy.IP].dst
        print(f"Pacote recebido de {pkt[scapy.IP].src} para {destination_ip}.")
        if pkt.haslayer(scapy.Raw):
            data = bytes(pkt[scapy.Raw])
            if data == b'EOF' and pkt[scapy.IP].src != '10.2.2.254':
                print("Imagem recebida com sucesso no roteador.")
                image_to_send = classify_image(f'router_images/reconstructed_image_{cont}.jpg', classes)               
                send_image(image_to_send, destination_ip)
                cont += 1
            else:
                if pkt[scapy.IP].src != '10.2.2.254':
                    with open(f'router_images/reconstructed_image_{cont}.jpg', 'ab') as f:
                        f.write(data)
                        print("Pacote recebido e escrito no arquivo no roteador.")

    elif pkt.haslayer(scapy.UDP) and pkt[scapy.UDP].dport == 12346:
        if pkt.haslayer(scapy.Raw):
            data = pkt[scapy.Raw].load
            blocked_classes = data.decode().split(',')
            print(f"Lista de classes recebida: {blocked_classes}")
            classes.extend(blocked_classes)


def send_image(image_path, dest_ip):
    print(dest_ip)
    with open(image_path, 'rb') as f:
        data = f.read()
    
    packets = [data[i:i+1024] for i in range(0, len(data), 1024)]
    
    for packet in packets:
        udp_packet = scapy.IP(dst=dest_ip)/scapy.UDP(sport=12345, dport=12345)/scapy.Raw(load=packet)
        scapy.send(udp_packet)
    
    eof_packet = scapy.IP(dst=dest_ip)/scapy.UDP(sport=12345, dport=12345)/scapy.Raw(load=b'EOF')
    scapy.send(eof_packet)
    
    print("Imagem enviada com sucesso para o destino final.")
    

def main(classes):
    interfaces = ['r1-eth1', 'r1-eth2']
    print("Aguardando pacotes no roteador...")
    scapy.sniff(iface=interfaces, prn=lambda pkt: handle_packet(pkt, classes), store=0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 router_in_the_middle.py <classes>")
        sys.exit(1)

    os.system('rm -rf router_images/*reconstructed*')

    classes = sys.argv[1].split(',')

    main(classes)