wget https://github.com/istio/istio/releases/download/1.18.2/istio-1.18.2-linux-amd64.tar.gz
tar -zxvf istio-1.18.2-linux-amd64.tar.gz
cd istio-1.18.2/
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
cd ..
rm istio-1.18.2-linux-amd64.tar.gz
rm -rf istio-1.18.2/
kubectl apply -f labeled_code.yaml
sleep 3
kubectl describe virtualservice reviews | grep "Route:
      Destination:
        Host:    reviews
        Subset:  v2
    Route:
      Destination:
        Host:    reviews
        Subset:  v1" && echo cloudeval_unit_test_passed