#!/usr/bin/env python3

import struct
import os
import sys
import json
import argparse
import ipaddress
from collections import defaultdict
from datetime import datetime

PORT_SERVICE_MAP = {
    # === WELL KNOWN PORTS ===
    20:   'FTP',       21:   'FTP',       22:   'SSH',
    23:   'TELNET',    25:   'SMTP',      53:   'DNS',
    67:   'DHCP',      68:   'DHCP',      69:   'TFTP',
    79:   'FINGER',    80:   'HTTP',      88:   'KERBEROS',
    110:  'POP3',      111:  'SUNRPC',    119:  'NNTP',
    123:  'NTP',       137:  'NETBIOS',   138:  'NETBIOS',
    139:  'SMB',       143:  'IMAP',      161:  'SNMP',
    162:  'SNMP',      179:  'BGP',       389:  'LDAP',
    443:  'HTTPS',     445:  'SMB',       465:  'SMTPS',
    500:  'ISAKMP',    514:  'SYSLOG',    515:  'IPP',
    520:  'RIP',       546:  'DHCP',      547:  'DHCP',
    548:  'AFP',       554:  'RTSP',      587:  'SMTP',
    631:  'IPP',       636:  'LDAPS',     853:  'DNS',
    873:  'RSYNC',     990:  'FTPS',      993:  'IMAP',
    995:  'POP3',
    # === REGISTERED PORTS ===
    1194: 'OPENVPN',  1433: 'MSSQL',    1723: 'PPTP',
    1883: 'MQTT',     1900: 'UPNP',     2049: 'NFS',
    3268: 'LDAP',     3306: 'MYSQL',    3389: 'RDP',
    3478: 'STUN',     3722: 'RDP',      4070: 'HTTP',
    5000: 'HTTP',     5060: 'SIP',      5222: 'XMPP',
    5223: 'XMPP',     5353: 'DNS',      5900: 'VNC',
    5938: 'TEAMVIEWER',
    6379: 'REDIS',    6881: 'BITTORRENT',
    7547: 'CWMP',     8080: 'HTTP',     8443: 'HTTPS',
    8883: 'MQTT',     9100: 'IPP',
    # === IoT / SMART HOME ===
    1400: 'SONOS',    1925: 'ROKU',     4444: 'HTTP',
    49152: 'UPNP',    55443: 'HTTPS',
}

PORT_RANGES = [
    (16384, 16403, 'RTP'),
    (49152, 65535, None),   # ephemeral — skip
]

def port_to_service(port: int) -> str | None:
    if port in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[port]
    for lo, hi, name in PORT_RANGES:
        if lo <= port <= hi:
            return name
    if port < 1024:
        return f'PORT{port}'
    return None


PCAP_MAGIC_LE = 0xA1B2C3D4
PCAP_MAGIC_BE = 0xD4C3B2A1
PCAP_NG_MAGIC = 0x0A0D0D0A

def read_pcap(filepath: str):
    with open(filepath, 'rb') as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError("File terlalu kecil")

        magic = struct.unpack('<I', hdr)[0]
        if magic == PCAP_NG_MAGIC:
            raise ValueError(
                "File adalah PCAP-NG format. Konversi dulu dengan:\n"
                "  editcap -F pcap input.pcapng output.pcap\n"
                "  atau: mergecap -F pcap -w output.pcap input.pcapng"
            )
        if magic not in (PCAP_MAGIC_LE, PCAP_MAGIC_BE):
            raise ValueError(f"Bukan file PCAP valid. Magic: {hex(magic)}")

        endian = '<' if magic == PCAP_MAGIC_LE else '>'
        rest = f.read(20)
        _, _, _, _, _, link_type = struct.unpack(f'{endian}HHIIII', rest)

        packets = []
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', rec)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            packets.append((ts_sec, data))

    return link_type, packets


def parse_ethernet(data: bytes):
    if len(data) < 14:
        return None
    dst = ':'.join(f'{b:02x}' for b in data[0:6])
    src = ':'.join(f'{b:02x}' for b in data[6:12])
    etype = struct.unpack('>H', data[12:14])[0]
    if etype == 0x8100 and len(data) >= 18:
        etype = struct.unpack('>H', data[16:18])[0]
        return dst, src, etype, data[18:]
    return dst, src, etype, data[14:]


def parse_ipv4(data: bytes):
    if len(data) < 20:
        return None
    ihl = (data[0] & 0x0F) * 4
    if ihl < 20 or len(data) < ihl:
        return None
    proto = data[9]
    src = '.'.join(str(b) for b in data[12:16])
    dst = '.'.join(str(b) for b in data[16:20])
    return src, dst, proto, data[ihl:]


def parse_transport(data: bytes):
    if len(data) < 4:
        return None, None
    sport, dport = struct.unpack('>HH', data[0:4])
    return sport, dport

def parse_tcp_flags(data: bytes) -> dict:
    if len(data) < 14:
        return {}
    flags_byte = data[13]
    return {
        'fin': bool(flags_byte & 0x01),
        'syn': bool(flags_byte & 0x02),
        'rst': bool(flags_byte & 0x04),
        'psh': bool(flags_byte & 0x08),
        'ack': bool(flags_byte & 0x10),
    }


def parse_arp(data: bytes):
    if len(data) < 28:
        return None, None
    sender_mac = ':'.join(f'{b:02x}' for b in data[8:14])
    sender_ip  = '.'.join(str(b) for b in data[14:18])
    return sender_mac, sender_ip


PRIVATE_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),
]

SKIP_IPS = {
    '0.0.0.0', '255.255.255.255',
    '224.0.0.1', '224.0.0.2', '224.0.0.251', '224.0.0.252',
    '239.255.255.250',
}
SKIP_MACS = {'ff:ff:ff:ff:ff:ff', '00:00:00:00:00:00', '01:00:5e'}

def is_private_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in PRIVATE_RANGES)
    except ValueError:
        return False

def is_skip_ip(ip_str: str) -> bool:
    if ip_str in SKIP_IPS:
        return True
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr.is_multicast or addr.is_unspecified or addr.is_reserved
    except ValueError:
        return True

def is_skip_mac(mac_str: str) -> bool:
    if mac_str in SKIP_MACS:
        return True
    try:
        first_byte = int(mac_str.split(':')[0], 16)
        return bool(first_byte & 0x01)
    except:
        return True

def auto_detect_subnets(hosts_dict: dict, arp_table: dict,
                        min_hosts: int = 1) -> list:
    bucket_hosts = defaultdict(set)

    all_ips = set(hosts_dict.keys()) | set(arp_table.keys())
    for ip in all_ips:
        if is_private_ip(ip) and not is_skip_ip(ip):
            try:
                net = ipaddress.ip_network(f'{ip}/24', strict=False)
                bucket_hosts[net].add(ip)
            except ValueError:
                pass

    detected = [net for net, hosts in bucket_hosts.items()
                if len(hosts) >= min_hosts]
    detected.sort(key=lambda n: len(bucket_hosts[n]), reverse=True)
    return detected


class HostProfiler:
    def __init__(self, subnet: str = None, min_packets: int = 3):
        if subnet:
            self.subnet_nets = [
                ipaddress.ip_network(s.strip(), strict=False)
                for s in subnet.split(',') if s.strip()
            ]
        else:
            self.subnet_nets = []
        self.min_packets = min_packets

        self.hosts = defaultdict(lambda: {
            'mac': None,
            'dst_well_known_ports': set(),
            'src_well_known_ports': set(),
            'pkt_count': 0,
            'first_seen': None,
            'last_seen': None,
            'peers_external': set(),
            'peers_internal': set(),
            'protocols': set(),
        })
        self.arp_table      = {}
        self.total_packets  = 0
        self._tcp_sessions  = {}
        self._tcp_established = set()

    def _is_internal(self, ip: str) -> bool:
        if is_skip_ip(ip):
            return False
        if self.subnet_nets:
            try:
                addr = ipaddress.ip_address(ip)
                return any(addr in net for net in self.subnet_nets)
            except ValueError:
                return False
        return is_private_ip(ip)

    def process_file(self, filepath: str):
        print(f"  [+] Parsing: {os.path.basename(filepath)}")
        link_type, packets = read_pcap(filepath)

        if link_type != 1:
            print(f"  [!] Link type {link_type} bukan Ethernet. Skip.")
            return

        for ts_sec, raw in packets:
            self.total_packets += 1
            eth = parse_ethernet(raw)
            if not eth:
                continue
            dst_mac, src_mac, ethertype, payload = eth

            if ethertype == 0x0806:
                sender_mac, sender_ip = parse_arp(payload)
                if sender_mac and sender_ip and not is_skip_ip(sender_ip):
                    self.arp_table[sender_ip] = sender_mac

            elif ethertype == 0x0800:
                ip_pkt = parse_ipv4(payload)
                if not ip_pkt:
                    continue
                src_ip, dst_ip, proto, l4 = ip_pkt

                if is_skip_ip(src_ip) or is_skip_ip(dst_ip):
                    continue
                if is_skip_mac(src_mac):
                    continue

                src_internal = self._is_internal(src_ip)
                dst_internal = self._is_internal(dst_ip)

                if not src_internal and not dst_internal:
                    continue

                if src_internal and not is_skip_mac(src_mac):
                    if src_ip not in self.arp_table:
                        self.arp_table[src_ip] = src_mac

                if src_internal:
                    h = self.hosts[src_ip]
                    if h['mac'] is None and not is_skip_mac(src_mac):
                        h['mac'] = src_mac
                    h['pkt_count'] += 1
                    if h['first_seen'] is None:
                        h['first_seen'] = ts_sec
                    h['last_seen'] = ts_sec
                    h['protocols'].add(proto)
                    if dst_internal:
                        h['peers_internal'].add(dst_ip)
                    else:
                        h['peers_external'].add(dst_ip)

                if proto in (6, 17):
                    sport, dport = parse_transport(l4)
                    if not sport or not dport:
                        continue

                    if proto == 6:
                        flags = parse_tcp_flags(l4)

                        if flags.get('syn') and not flags.get('ack'):
                            sess_key = (src_ip, dst_ip, sport, dport)
                            self._tcp_sessions[sess_key] = 'syn_sent'

                        elif flags.get('syn') and flags.get('ack'):
                            sess_key = (dst_ip, src_ip, dport, sport)
                            if sess_key in self._tcp_sessions:
                                self._tcp_established.add((src_ip, sport))
                                svc = port_to_service(sport)
                                if svc and dst_internal:
                                    self.hosts[src_ip]['src_well_known_ports'].add((sport, svc))
                                if svc and src_internal:
                                    self.hosts[dst_ip]['dst_well_known_ports'].add((sport, svc))

                        elif flags.get('rst'):
                            pass

                        elif flags.get('psh') or (flags.get('ack') and not flags.get('syn')):
                            sess_key_rev = (dst_ip, src_ip, dport, sport)
                            if sess_key_rev in self._tcp_sessions:
                                confirmed = (src_ip, sport)
                                if confirmed not in self._tcp_established:
                                    self._tcp_established.add(confirmed)
                                    svc = port_to_service(sport)
                                    if svc and src_internal:
                                        self.hosts[src_ip]['src_well_known_ports'].add((sport, svc))

                    else:
                        if dst_internal:
                            svc = port_to_service(dport)
                            if svc:
                                self.hosts[dst_ip]['src_well_known_ports'].add((dport, svc))
                        if src_internal:
                            svc = port_to_service(dport)
                            if svc:
                                self.hosts[src_ip]['dst_well_known_ports'].add((dport, svc))

        print(f"     Packets processed: {len(packets)}")

    def build_profiles(self, subnet: str = None):
        for ip, info in self.hosts.items():
            if info['mac'] is None and ip in self.arp_table:
                info['mac'] = self.arp_table[ip]

        if not self.subnet_nets:
            if subnet:
                self.subnet_nets = [
                    ipaddress.ip_network(s.strip(), strict=False)
                    for s in subnet.split(',') if s.strip()
                ]
            else:
                detected = auto_detect_subnets(self.hosts, self.arp_table, min_hosts=1)
                if detected:
                    self.subnet_nets = detected
                    labels = ', '.join(str(n) for n in detected)
                    print(f"  [auto] {len(detected)} subnet terdeteksi: {labels}")
                else:
                    print("  [warn] Tidak ada subnet terdeteksi, memakai semua private IP")

        profiles = []
        skipped  = []

        for ip, info in sorted(self.hosts.items()):
            if not self._is_internal(ip):
                continue

            mac = info['mac'] or self.arp_table.get(ip)
            if not mac or is_skip_mac(mac):
                skipped.append((ip, 'no valid MAC'))
                continue

            if info['pkt_count'] < self.min_packets:
                skipped.append((ip, f"pkt_count={info['pkt_count']} < min={self.min_packets}"))
                continue

            oui = mac[:8]

            svc_set = set()
            for port, svc_name in info['dst_well_known_ports']:
                svc_set.add(svc_name)
            for port, svc_name in info['src_well_known_ports']:
                svc_set.add(svc_name)

            if not svc_set:
                skipped.append((ip, 'no recognizable services'))
                continue

            profiles.append({
                'ip':          ip,
                'oui':         oui,
                'mac':         mac,
                'svc':         sorted(svc_set),
                'pkt_count':   info['pkt_count'],
                'proto_codes': sorted(info['protocols']),
                'peers_ext':   len(info['peers_external']),
                'peers_int':   len(info['peers_internal']),
                'first_seen':  info['first_seen'],
                'last_seen':   info['last_seen'],
            })

        return profiles, skipped


def to_notebook_format(profiles: list) -> list:
    return [
        {'oui': p['oui'], 'svc': tuple(p['svc'])}
        for p in profiles
    ]

def print_notebook_snippet(profiles_notebook: list):
    print("\n" + "="*60)
    print("# FORMAT SIAP PASTE KE NOTEBOOK COLAB")
    print("# Masukkan ini ke variabel 'dataset'")
    print("="*60)
    print("dataset = [")
    for p in profiles_notebook:
        print(f"    {p},")
    print("]")

def print_summary(profiles: list, skipped: list):
    print(f"\n{'='*60}")
    print(f"RINGKASAN HASIL EKSTRAKSI")
    print(f"{'='*60}")
    print(f"Host profil berhasil diekstrak: {len(profiles)}")
    print(f"Host dilewati: {len(skipped)}")
    if skipped:
        print(f"\nDilewati:")
        for ip, reason in skipped:
            print(f"  {ip:20s} → {reason}")
    print(f"\nDetail profil:")
    for p in profiles:
        dur = (p['last_seen'] - p['first_seen']) if p['first_seen'] and p['last_seen'] else 0
        protos = {1:'ICMP', 6:'TCP', 17:'UDP'}.get
        proto_names = [protos(c, f'P{c}') for c in p['proto_codes']]
        print(f"\n  IP  : {p['ip']}")
        print(f"  MAC : {p['mac']}  (OUI: {p['oui']})")
        print(f"  Pkts: {p['pkt_count']}  Duration: {dur}s")
        print(f"  Proto: {proto_names}  Peers ext: {p['peers_ext']}  int: {p['peers_int']}")
        print(f"  Services ({len(p['svc'])}): {p['svc']}")


def collect_pcap_files(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for root, dirs, fnames in os.walk(path):
            for fn in fnames:
                if fn.lower().endswith(('.pcap', '.cap')):
                    files.append(os.path.join(root, fn))
        return sorted(files)
    raise FileNotFoundError(f"Path tidak ditemukan: {path}")


def run(pcap_path: str, subnet: str = None, output: str = None,
        min_packets: int = 3, merge_mode: str = 'file', label: str = None,
        exclude_oui: str = None):
    files = collect_pcap_files(pcap_path)
    print(f"\nDitemukan {len(files)} file PCAP")

    all_profiles_notebook = []
    all_profiles_detail   = []

    if merge_mode == 'file':
        for fpath in files:
            print(f"\n[FILE] {os.path.basename(fpath)}")
            profiler = HostProfiler(subnet=subnet, min_packets=min_packets)
            try:
                profiler.process_file(fpath)
            except ValueError as e:
                print(f"  [SKIP] {e}")
                continue

            profiles, skipped = profiler.build_profiles()
            if not profiles:
                print(f"  [!] Tidak ada host valid ditemukan")
                continue

            all_profiles_detail.extend(profiles)
            all_profiles_notebook.extend(to_notebook_format(profiles))
            print_summary(profiles, skipped)

    else:
        profiler = HostProfiler(subnet=subnet, min_packets=min_packets)
        for fpath in files:
            try:
                profiler.process_file(fpath)
            except ValueError as e:
                print(f"  [SKIP] {fpath}: {e}")
        profiles, skipped = profiler.build_profiles()
        all_profiles_detail.extend(profiles)
        all_profiles_notebook.extend(to_notebook_format(profiles))
        print_summary(profiles, skipped)

    if exclude_oui:
        excluded = {o.strip().lower() for o in exclude_oui.split(',') if o.strip()}
        before = len(all_profiles_detail)
        all_profiles_detail   = [p for p in all_profiles_detail
                                  if p['oui'].lower() not in excluded]
        all_profiles_notebook = [p for p in all_profiles_notebook
                                  if p['oui'].lower() not in excluded]
        removed = before - len(all_profiles_detail)
        if removed:
            print(f"  [exclude-oui] {removed} host dengan OUI {excluded} dikecualikan")

    print_notebook_snippet(all_profiles_notebook)

    result = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_files': [os.path.basename(f) for f in files],
            'total_hosts': len(all_profiles_detail),
            'subnet': subnet or 'auto-detected',
            'min_packets_threshold': min_packets,
            'label': label or '',
        },
        'profiles_detail':   all_profiles_detail,
        'profiles_notebook': all_profiles_notebook,
    }

    if output:
        safe = json.dumps(result, indent=2, default=list)
        with open(output, 'w') as f:
            f.write(safe)
        print(f"\n[+] Output disimpan ke: {output}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PCAP → Host Profiler (IoT-23 / CTU format)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # 1 file, auto-detect subnet
  python3 pcap_host_profiler.py --pcap capture.pcap

  # Direktori berisi banyak PCAP, subnet eksplisit
  python3 pcap_host_profiler.py --pcap ./iot23/ --subnet 192.168.1.0/24

  # Simpan hasil ke JSON
  python3 pcap_host_profiler.py --pcap capture.pcap --output hasil.json

  # Gabung semua file (profiling per-IP lintas file)
  python3 pcap_host_profiler.py --pcap ./iot23/ --merge all

  # Filter: host harus punya minimal 10 paket
  python3 pcap_host_profiler.py --pcap capture.pcap --min-pkts 10
        """
    )
    parser.add_argument('--pcap',     required=True,
                        help='Path ke file .pcap atau direktori')
    parser.add_argument('--subnet',   default=None,
                        help='Subnet internal. Satu: 192.168.1.0/24, atau comma-separated\n'
                             '                             untuk multi-subnet: 192.168.1.0/24,10.0.0.0/8\n'
                             '                             (default: auto-detect semua subnet aktif)')
    parser.add_argument('--output',   default=None,
                        help='Path output JSON (opsional)')
    parser.add_argument('--min-pkts', type=int, default=3,
                        help='Min paket per host (default: 3)')
    parser.add_argument('--merge',    choices=['file', 'all'], default='file',
                        help='file=1 cluster/file, all=gabung semua (default: file)')
    parser.add_argument('--exclude-oui', default=None,
                        help='Comma-separated OUI yang dikecualikan dari profiling\n'
                             'Contoh: ca:fe:00 untuk exclude scanner host\n'
                             'Format: xx:xx:xx,yy:yy:yy')
    parser.add_argument('--label',    default=None,
                        help='Label device type, disimpan ke metadata JSON')

    args = parser.parse_args()
    run(
        pcap_path=args.pcap,
        subnet=args.subnet,
        output=args.output,
        min_packets=args.min_pkts,
        merge_mode=args.merge,
        label=args.label,
        exclude_oui=args.exclude_oui,
    )
