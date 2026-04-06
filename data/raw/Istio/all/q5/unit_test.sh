wget https://github.com/istio/istio/releases/download/1.18.2/istio-1.18.2-linux-amd64.tar.gz
tar -zxvf istio-1.18.2-linux-amd64.tar.gz
cd istio-1.18.2/
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
cd ..
rm istio-1.18.2-linux-amd64.tar.gz
rm -rf istio-1.18.2/
kubectl create -f labeled_code.yaml
sleep 5
kubectl describe gateways.networking.istio.io my-gateway | grep "Hosts:
      ns1/*
      ns2/foo.bar.com
    Port:
      Name:      http
      Number:    80
      Protocol:  HTTP" && echo cloudeval_unit_test_passed