wget https://github.com/istio/istio/releases/download/1.18.2/istio-1.18.2-linux-amd64.tar.gz
tar -zxvf istio-1.18.2-linux-amd64.tar.gz
cd istio-1.18.2/
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
cd ..
rm istio-1.18.2-linux-amd64.tar.gz
rm -rf istio-1.18.2/

echo "apiVersion: v1
kind: Service
metadata:
  name: tornado
  labels:
    app: tornado
    service: tornado
spec:
  ports:
  - port: 8888
    name: http
  selector:
    app: tornado
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tornado
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tornado
      version: v1
  template:
    metadata:
      labels:
        app: tornado
        version: v1
    spec:
      containers:
      - name: tornado
        image: hiroakis/tornado-websocket-example
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8888
" | kubectl create -f -
echo "apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: tornado-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - \"*\"
" | kubectl create -f -
kubectl create -f labeled_code.yaml
sleep 30

kubectl describe virtualservice tornado | grep "Gateways:
    tornado-gateway" && echo cloudeval_unit_test_passed