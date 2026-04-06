minikube addons enable ingress
echo "apiVersion: apps/v1
kind: Deployment
metadata:
  name: tomcat-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tomcat
  template:
    metadata:
      labels:
        app: tomcat
    spec:
      containers:
      - name: tomcat
        image: tomcat:latest
        ports:
        - containerPort: 8080
" | kubectl apply -f -
sleep 10
echo "apiVersion: v1
kind: Service
metadata:
  name: service1
spec:
  selector:
    app: tomcat  # This should match the labels on your Tomcat pods
  ports:
    - protocol: TCP
      port: 8080  # This is the port the service will listen on
      targetPort: 8080  # This is the port on the pod that traffic will be sent to
" | kubectl apply -f -
sleep 5
echo "apiVersion: v1
kind: Service
metadata:
  name: service2
spec:
  selector:
    app: tomcat  # This should match the labels on your Tomcat pods
  ports:
    - protocol: TCP
      port: 8080  # This is the port the service will listen on
      targetPort: 8080  # This is the port on the pod that traffic will be sent to
" | kubectl apply -f -
sleep 15
kubectl apply -f labeled_code.yaml
sleep 15
minikube kubectl describe ingress ingress-wildcard-host | egrep "Scheduled for sync" && echo cloudeval_unit_test_passed
