from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info
from mininet.cli import CLI

class AdvancedTopo(Topo):
    "One router with two hosts"

    def build(self, **_opts):
        # Create router
        r1 = self.addHost('r1', ip=None)

        # Create hosts
        h1 = self.addHost('h1', ip=None, defaultRoute='via 10.1.1.254')
        h2 = self.addHost('h2', ip=None, defaultRoute='via 10.2.2.254')
        
        # Host to Router 1
        self.addLink(h1, r1, intfName1='h1-eth0', params1={'ip': '10.1.1.1/24'},
                     intfName2='r1-eth1', params2={'ip': '10.1.1.254/24'})
        
        # Host to Router 1
        self.addLink(h2, r1, intfName1='h2-eth0', params1={'ip': '10.2.2.1/24'},
                     intfName2='r1-eth2', params2={'ip': '10.2.2.254/24'})

def run():
    "Advanced topology with one router and route exchange testing"
    net = Mininet(topo=AdvancedTopo(), controller=None)
    
    for _, v in net.nameToNode.items():
        for itf in v.intfList():
            v.cmd('ethtool -K ' + itf.name + ' tx off rx off')

    net['r1'].cmd('sysctl -w net.ipv4.ip_forward=0')

    net['r1'].cmd('ip route add 10.1.1.0/24 dev r1-eth1')
    net['r1'].cmd('ip route add 10.2.2.0/24 dev r1-eth2')
    
    net.start()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()