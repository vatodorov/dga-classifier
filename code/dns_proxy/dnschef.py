#!/usr/bin/env python3

# Manticore
# This is a modified version of the DNS Chef


# DNSChef is a highly configurable DNS Proxy for Penetration Testers 
# and Malware Analysts. Please visit http://thesprawl.org/projects/dnschef/
# for the latest version and documentation. Please forward all issues and
# concerns to iphelix [at] thesprawl.org.

DNSCHEF_VERSION = "0.4"

# Copyright (C) 2019 Peter Kacherginsky, Marcello Salvati
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without 
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from dnslib import *

from argparse import ArgumentParser
from configparser import ConfigParser

import binascii
import logging
import operator
import requests
import random
import socketserver
import socket
import sys
import threading


class DNSChefFormatter(logging.Formatter):

    FORMATS = {
        logging.ERROR: "(%(asctime)s) [!] %(msg)s",
        logging.INFO: "(%(asctime)s) [*] %(msg)s",
        logging.WARNING: "WARNING: %(msg)s",
        logging.DEBUG: "DBG: %(module)s: %(lineno)d: %(msg)s",
        "DEFAULT": "%(asctime)s - %(msg)s"
    }

    def format(self, record):
        format_orig = self._style._fmt

        self._style._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        result = logging.Formatter.format(self, record)

        self._style._fmt = format_orig

        return result

log = logging.getLogger("dnschef")
log.setLevel(logging.DEBUG)

log_ch = logging.StreamHandler()
log_ch.setLevel(logging.INFO)
log_ch.setFormatter(DNSChefFormatter(datefmt="%H:%M:%S"))
log.addHandler(log_ch)

# DNSHandler Mixin. The class contains generic functions to parse DNS requests and
# calculate an appropriate response based on user parameters.
class DNSHandler():

    def parse(self, data):
        response = ""

        try:
            # Parse data as DNS
            d = DNSRecord.parse(data)

        except Exception:
            log.error(f"{self.client_address[0]}: ERROR: invalid DNS request")

        else:
            # Only Process DNS Queries
            if QR[d.header.qr] == "QUERY":

                # Gather query parameters
                # NOTE: Do not lowercase qname here, because we want to see
                #       any case request weirdness in the logs.
                qname = str(d.q.qname)

                # Chop off the last period
                if qname[-1] == '.': qname = qname[:-1]
                qtype = QTYPE[d.q.qtype]

                # Proxy the request
                log.info(f"{self.client_address[0]}: proxying the response of type '{qtype}' for {qname}")

                nameserver_tuple = random.choice(self.server.nameservers).split('#')
                response = self.proxyrequest(data, *nameserver_tuple)

                log.info('Running the DGA scoring algorithm for {}'.format(qname))

                # Executes the DGA scoring
                resp = self.dga_score(
                    qname=qname,
                    appserver=appserver,
                    appserverport=appserverport,
                    username=username,
                    password=password
                )

                # Decide which IP to return based on the cutoff
                if resp.json()['data'][0]['dga_score'] > float(dgaprobcutoff):
                    # Creates the DNS response
                    reply = self.build_dns_response(
                        qname=qname,
                        dns_header_id=d.header.id,
                        ip=redirecturl
                    )

                    # Converts the DNS response to binary
                    response = reply.pack()

        return response

    def dga_score(self, qname, appserver, appserverport, username, password):
        """
        Runs the DGA scoring on Manticore

        :param qname:
        :param appserver:
        :param appserverport:
        :param username:
        :param password:
        :return:
        """

        ep = '/api/predict'

        resp = requests.post(
            url='http://{}:{}{}'.format(appserver, appserverport, ep),
            json={'domain': qname},
            auth=(username, password)
        )

        return resp

    def build_dns_response(self, qname, dns_header_id, ip):
        """
        Function to build the DNS response

        :param qname str: QNAME in the DNS package
        :param dns_header_id str: DNS header ID
        :param ip str: The IP Address the request would return
        :return:
        """

        resp = DNSRecord(
            DNSHeader(qr=1, aa=1, ra=1, id=dns_header_id),
            q=DNSQuestion(qname),
            a=RR(qname, rdata=A(ip))
        )

        return resp

    # Find appropriate ip address to use for a queried name. The function can
    def findnametodns(self, qname, nametodns):

        # Make qname case insensitive
        qname = qname.lower()

        # Split and reverse qname into components for matching.
        qnamelist = qname.split('.')
        qnamelist.reverse()

        # HACK: It is important to search the nametodns dictionary before iterating it so that
        # global matching ['*.*.*.*.*.*.*.*.*.*'] will match last. Use sorting for that.
        for domain, host in sorted(iter(nametodns.items()), key=operator.itemgetter(1)):

            # NOTE: It is assumed that domain name was already lowercased
            #       when it was loaded through --file, --fakedomains or --truedomains
            #       don't want to waste time lowercasing domains on every request.

            # Split and reverse domain into components for matching
            domain = domain.split('.')
            domain.reverse()

            # Compare domains in reverse.
            for a, b in zip(qnamelist, domain):
                if a != b and b != "*":
                    break
            else:
                # Could be a real IP or False if we are doing reverse matching with 'truedomains'
                return host
        else:
            return False

    # Obtain a response from a real DNS server.
    def proxyrequest(self, request, host, port="53", protocol="udp"):
        reply = None
        try:
            if self.server.ipv6:
                if protocol == "udp":
                    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                elif protocol == "tcp":
                    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            else:
                if protocol == "udp":
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                elif protocol == "tcp":
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            sock.settimeout(3.0)

            # Send the proxy request to a randomly chosen DNS server
            if protocol == "udp":
                sock.sendto(request, (host, int(port)))
                reply = sock.recv(1024)
                sock.close()

            elif protocol == "tcp":
                sock.connect((host, int(port)))

                # Add length for the TCP request
                length = binascii.unhexlify("%04x" % len(request))
                sock.sendall(length+request)

                # Strip length from the response
                reply = sock.recv(1024)
                reply = reply[2:]

                sock.close()

        except Exception as e:
            log.error(f"[!] Could not proxy request: {e}")
        else:
            return reply

# UDP DNS Handler for incoming requests
class UDPHandler(DNSHandler, socketserver.BaseRequestHandler):

    def handle(self):
        (data, socket) = self.request
        response = self.parse(data)

        if response:
            socket.sendto(response, self.client_address)

# TCP DNS Handler for incoming requests
class TCPHandler(DNSHandler, socketserver.BaseRequestHandler):

    def handle(self):
        data = self.request.recv(1024)

        # Remove the addition "length" parameter used in the
        # TCP DNS protocol
        data = data[2:]
        response = self.parse(data)

        if response:
            # Calculate and add the additional "length" parameter
            # used in TCP DNS protocol 
            length = binascii.unhexlify("%04x" % len(response))
            self.request.sendall(length + response)

class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):

    # Override SocketServer.UDPServer to add extra parameters
    def __init__(self, server_address, RequestHandlerClass, nametodns, nameservers, ipv6, log):
        self.nametodns  = nametodns
        self.nameservers = nameservers
        self.ipv6        = ipv6
        self.address_family = socket.AF_INET6 if self.ipv6 else socket.AF_INET
        self.log = log

        socketserver.UDPServer.__init__(self, server_address, RequestHandlerClass)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):

    # Override default value
    allow_reuse_address = True

    # Override SocketServer.TCPServer to add extra parameters
    def __init__(self, server_address, RequestHandlerClass, nametodns, nameservers, ipv6, log):
        self.nametodns   = nametodns
        self.nameservers = nameservers
        self.ipv6        = ipv6
        self.address_family = socket.AF_INET6 if self.ipv6 else socket.AF_INET
        self.log = log

        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass)

# Initialize and start the DNS Server
def start_cooking(interface, nametodns, nameservers, tcp=False, ipv6=False, port="53", logfile=None):
    try:

        if logfile:
            fh = logging.FileHandler(logfile, encoding='UTF-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(DNSChefFormatter(datefmt="%d/%b/%Y:%H:%M:%S %z"))
            log.addHandler(fh)

            log.info("DNSChef is active.")

        if tcp:
            log.info("DNSChef is running in TCP mode")
            server = ThreadedTCPServer((interface, int(port)), TCPHandler, nametodns, nameservers, ipv6, log)
        else:
            server = ThreadedUDPServer((interface, int(port)), UDPHandler, nametodns, nameservers, ipv6, log)

        # Start a thread with the server -- that thread will then start
        # more threads for each request
        server_thread = threading.Thread(target=server.serve_forever)

        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()

        # Loop in the main thread
        while True: time.sleep(100)

    except (KeyboardInterrupt, SystemExit):
        server.shutdown()
        log.info("DNSChef is shutting down.")
        sys.exit()

    except Exception as e:
        log.error(f"Failed to start the server: {e}")


if __name__ == "__main__":

    header  = "          _                _          __  \n"
    header += "         | | version %s  | |        / _| \n" % DNSCHEF_VERSION
    header += "       __| |_ __  ___  ___| |__   ___| |_ \n"
    header += "      / _` | '_ \/ __|/ __| '_ \ / _ \  _|\n"
    header += "     | (_| | | | \__ \ (__| | | |  __/ |  \n"
    header += "      \__,_|_| |_|___/\___|_| |_|\___|_|  \n"
    header += "                   iphelix@thesprawl.org  \n"

    # Parse command line arguments
    parser = ArgumentParser(usage = "dnschef.py [options]:\n" + header, description="DNSChef is a highly configurable DNS Proxy for Penetration Testers and Malware Analysts. It is capable of fine configuration of which DNS replies to modify or to simply proxy with real responses. In order to take advantage of the tool you must either manually configure or poison DNS server entry to point to DNSChef. The tool requires root privileges to run on privileged ports." )

    fakegroup = parser.add_argument_group("Fake DNS records:")
    fakegroup.add_argument('--fakeip', metavar="192.0.2.1", help='IP address to use for matching DNS queries. If you use this parameter without specifying domain names, then all \'A\' queries will be spoofed. Consider using --file argument if you need to define more than one IP address.')
    fakegroup.add_argument('--fakeipv6', metavar="2001:db8::1", help='IPv6 address to use for matching DNS queries. If you use this parameter without specifying domain names, then all \'AAAA\' queries will be spoofed. Consider using --file argument if you need to define more than one IPv6 address.')
    fakegroup.add_argument('--fakemail', metavar="mail.fake.com", help='MX name to use for matching DNS queries. If you use this parameter without specifying domain names, then all \'MX\' queries will be spoofed. Consider using --file argument if you need to define more than one MX record.')
    fakegroup.add_argument('--fakealias', metavar="www.fake.com", help='CNAME name to use for matching DNS queries. If you use this parameter without specifying domain names, then all \'CNAME\' queries will be spoofed. Consider using --file argument if you need to define more than one CNAME record.')
    fakegroup.add_argument('--fakens', metavar="ns.fake.com", help='NS name to use for matching DNS queries. If you use this parameter without specifying domain names, then all \'NS\' queries will be spoofed. Consider using --file argument if you need to define more than one NS record.')
    fakegroup.add_argument('--file', help="Specify a file containing a list of DOMAIN=IP pairs (one pair per line) used for DNS responses. For example: google.com=1.1.1.1 will force all queries to 'google.com' to be resolved to '1.1.1.1'. IPv6 addresses will be automatically detected. You can be even more specific by combining --file with other arguments. However, data obtained from the file will take precedence over others.")

    mexclusivegroup = parser.add_mutually_exclusive_group()
    mexclusivegroup.add_argument('--fakedomains', metavar="thesprawl.org,google.com", help='A comma separated list of domain names which will be resolved to FAKE values specified in the the above parameters. All other domain names will be resolved to their true values.')
    mexclusivegroup.add_argument('--truedomains', metavar="thesprawl.org,google.com", help='A comma separated list of domain names which will be resolved to their TRUE values. All other domain names will be resolved to fake values specified in the above parameters.')

    rungroup = parser.add_argument_group("Optional runtime parameters.")
    rungroup.add_argument("--logfile", metavar="FILE", help="Specify a log file to record all activity")
    rungroup.add_argument("--nameservers", metavar="8.8.8.8#53 or 4.2.2.1#53#tcp or 2001:4860:4860::8888", default='8.8.8.8', help='A comma separated list of alternative DNS servers to use with proxied requests. Nameservers can have either IP or IP#PORT format. A randomly selected server from the list will be used for proxy requests when provided with multiple servers. By default, the tool uses Google\'s public DNS server 8.8.8.8 when running in IPv4 mode and 2001:4860:4860::8888 when running in IPv6 mode.')
    rungroup.add_argument("-i","--interface", metavar="127.0.0.1 or ::1", default="127.0.0.1", help='Define an interface to use for the DNS listener. By default, the tool uses 127.0.0.1 for IPv4 mode and ::1 for IPv6 mode.')
    rungroup.add_argument("-t","--tcp", action="store_true", default=False, help="Use TCP DNS proxy instead of the default UDP.")
    rungroup.add_argument("-6","--ipv6", action="store_true", default=False, help="Run in IPv6 mode.")
    rungroup.add_argument("-p","--port", metavar="53", default="53", help='Port number to listen for DNS requests.')
    rungroup.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=True, help="Don't show headers.")


    # ============================== Added for Manticore ======================================= #

    rungroup.add_argument("--appserver", help="The IP address of the server the scoring app is running on.")
    rungroup.add_argument("--appserverport", default="5001", help="The port the scoring app is running on.")
    rungroup.add_argument("--username", help="Username to authenticate with Manticore")
    rungroup.add_argument("--password", help="Password to authenticate with Manticore")
    rungroup.add_argument("--dgaprobcutoff", help="The probability score cutoff to determine which domains are DGAs.")
    rungroup.add_argument("--redirecturl", default="www.google.com", help="The URL a user will be redirected to, if the DGA score for a domain is above the cutoff")

    options = parser.parse_args()

    # Get from the arguments the server IP and the port the scoring app is running on
    if options.appserver:
        appserver = options.appserver
    else:
        sys.exit('Please provide an IP address for the Manticore server')

    if options.appserverport:
        appserverport = options.appserverport

    log.info('The DNS proxy will use the following DGA scoring server and port: {}:{}'.format(appserver, appserverport))

    # Get the authentication parameters for Manticore
    if options.username:
        username = options.username
    else:
        sys.exit('Please provide a username for logging to Manticore.')

    if options.password:
        password = options.password
    else:
        sys.exit('Please provide a password to authenticate with the Manticore server.')

    # Get the probability cutoff for the DGA prediction
    #   If the Probability is above the cutoff, the domain is DGA, otherwise not
    if options.dgaprobcutoff:
        dgaprobcutoff = options.dgaprobcutoff

    # Redirect page
    redirecturl = options.redirecturl

    # ========================================================================================== #

    # Print program header
    if options.verbose:
        print(header)

    # Main storage of domain filters
    # NOTE: RDMAP is a dictionary map of qtype strings to handling classes
    nametodns = dict()
    for qtype in list(RDMAP.keys()):
        nametodns[qtype] = dict()

    if not (options.fakeip or options.fakeipv6) and (options.fakedomains or options.truedomains):
        log.error("You have forgotten to specify which IP to use for fake responses")
        sys.exit(0)

    # Notify user about alternative listening port
    if options.port != "53":
        log.info(f"Listening on an alternative port {options.port}")

    # Adjust defaults for IPv6
    if options.ipv6:
        log.info("Using IPv6 mode.")
        if options.interface == "127.0.0.1":
            options.interface = "::1"

        if options.nameservers == "8.8.8.8":
            options.nameservers = "2001:4860:4860::8888"

    log.info(f"DNSChef started on interface: {options.interface}")

    # Use alternative DNS servers
    if options.nameservers:
        nameservers = options.nameservers.split(',')
        log.info(f"Using the following nameservers: {', '.join(nameservers)}")

    # External file definitions
    if options.file:
        config = ConfigParser()
        config.read(options.file)
        for section in config.sections():

            if section in nametodns:
                for domain, record in config.items(section):

                    # Make domain case insensitive
                    domain = domain.lower()

                    nametodns[section][domain] = record
                    log.info(f"Cooking {section} replies for domain {domain} with '{record}'")
            else:
                log.warning(f"DNS Record '{section}' is not supported. Ignoring section contents.")

    # DNS Record and Domain Name definitions
    # NOTE: '*.*.*.*.*.*.*.*.*.*' domain is used to match all possible queries.
    if options.fakeip or options.fakeipv6 or options.fakemail or options.fakealias or options.fakens:
        fakeip     = options.fakeip
        fakeipv6   = options.fakeipv6
        fakemail   = options.fakemail
        fakealias  = options.fakealias
        fakens     = options.fakens

        if options.fakedomains:
            for domain in options.fakedomains.split(','):

                # Make domain case insensitive
                domain = domain.lower()
                domain = domain.strip()

                if fakeip:
                    nametodns["A"][domain] = fakeip
                    log.info(f"Cooking A replies to point to {options.fakeip} matching: {domain}")

                if fakeipv6:
                    nametodns["AAAA"][domain] = fakeipv6
                    log.info(f"Cooking AAAA replies to point to {options.fakeipv6} matching: {domain}")

                if fakemail:
                    nametodns["MX"][domain] = fakemail
                    log.info(f"Cooking MX replies to point to {options.fakemail} matching: {domain}")

                if fakealias:
                    nametodns["CNAME"][domain] = fakealias
                    log.info(f"Cooking CNAME replies to point to {options.fakealias} matching: {domain}")

                if fakens:
                    nametodns["NS"][domain] = fakens
                    log.info(f"Cooking NS replies to point to {options.fakens} matching: {domain}")

        elif options.truedomains:
            for domain in options.truedomains.split(','):

                # Make domain case insensitive
                domain = domain.lower()
                domain = domain.strip()

                if fakeip:
                    nametodns["A"][domain] = False
                    log.info(f"Cooking A replies to point to {options.fakeip} not matching: {domain}")
                    nametodns["A"]['*.*.*.*.*.*.*.*.*.*'] = fakeip

                if fakeipv6:
                    nametodns["AAAA"][domain] = False
                    log.info(f"Cooking AAAA replies to point to {options.fakeipv6} not matching: {domain}")
                    nametodns["AAAA"]['*.*.*.*.*.*.*.*.*.*'] = fakeipv6

                if fakemail:
                    nametodns["MX"][domain] = False
                    log.info(f"Cooking MX replies to point to {options.fakemail} not matching: {domain}")
                    nametodns["MX"]['*.*.*.*.*.*.*.*.*.*'] = fakemail

                if fakealias:
                    nametodns["CNAME"][domain] = False
                    log.info(f"Cooking CNAME replies to point to {options.fakealias} not matching: {domain}")
                    nametodns["CNAME"]['*.*.*.*.*.*.*.*.*.*'] = fakealias

                if fakens:
                    nametodns["NS"][domain] = False
                    log.info(f"Cooking NS replies to point to {options.fakens} not matching: {domain}")
                    nametodns["NS"]['*.*.*.*.*.*.*.*.*.*'] = fakealias

        else:

            # NOTE: '*.*.*.*.*.*.*.*.*.*' domain is a special ANY domain
            #       which is compatible with the wildflag algorithm above.

            if fakeip:
                nametodns["A"]['*.*.*.*.*.*.*.*.*.*'] = fakeip
                log.info(f"Cooking all A replies to point to {fakeip}")

            if fakeipv6:
                nametodns["AAAA"]['*.*.*.*.*.*.*.*.*.*'] = fakeipv6
                log.info(f"Cooking all AAAA replies to point to {fakeipv6}")

            if fakemail:
                nametodns["MX"]['*.*.*.*.*.*.*.*.*.*'] = fakemail
                log.info(f"Cooking all MX replies to point to {fakemail}")

            if fakealias:
                nametodns["CNAME"]['*.*.*.*.*.*.*.*.*.*'] = fakealias
                log.info(f"Cooking all CNAME replies to point to {fakealias}")

            if fakens:
                nametodns["NS"]['*.*.*.*.*.*.*.*.*.*'] = fakens
                log.info(f"Cooking all NS replies to point to {fakens}")

    # Proxy all DNS requests
    if not options.fakeip and not options.fakeipv6 and not options.fakemail and not options.fakealias and not options.fakens and not options.file:
        log.info("No parameters were specified. Running in full proxy mode")

    log.info('The proxy will use {}:{} for running the DGA scoring'.format(appserver, appserverport))
    log.info('The user provided DGA probability cutoff is {}'.format(dgaprobcutoff))

    # Launch DNSChef
    start_cooking(
        interface=options.interface,
        nametodns=nametodns,
        nameservers=nameservers,
        tcp=options.tcp,
        ipv6=options.ipv6,
        port=options.port,
        logfile=options.logfile
    )
