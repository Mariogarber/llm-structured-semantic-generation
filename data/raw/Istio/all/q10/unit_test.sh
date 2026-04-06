wget https://github.com/istio/istio/releases/download/1.18.2/istio-1.18.2-linux-amd64.tar.gz
tar -zxvf istio-1.18.2-linux-amd64.tar.gz
cd istio-1.18.2/
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
cd ..
rm istio-1.18.2-linux-amd64.tar.gz
rm -rf istio-1.18.2/

kubectl apply -f https://k8s.io/examples/pods/simple-pod.yaml
istioctl install --set profile=demo -y
kubectl apply -f labeled_code.yaml
sleep 5
kubectl describe serviceentry github-https | grep  "Hosts:
    github.com" && echo cloudeval_unit_test_passed