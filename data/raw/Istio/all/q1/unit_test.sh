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
kubectl describe destinationrule bookinfo-ratings | grep "Host:  ratings.prod.svc.cluster.local
  Traffic Policy:
    Load Balancer:
      Simple:  LEAST_REQUEST" && echo cloudeval_unit_test_passed