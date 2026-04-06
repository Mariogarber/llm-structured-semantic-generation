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
  name: tomcat-svc
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
sleep 40
kubectl get svc
# This is a case when you need reg-match, please refer to 
minikube kubectl describe ingress tomcat-https | egrep "tomcat-svc:8080\ \(([[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3})\:8080\)" && echo cloudeval_unit_test_passed
# INCLUDE: regex like "tomcat-svc:8080 (10.244.0.3:8080)", 10.244.0.3:8080
