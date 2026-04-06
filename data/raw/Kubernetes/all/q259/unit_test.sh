echo "apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
spec:
  rules:
  - host: example.com" | kubectl apply -f -
sleep 15
kubectl apply -f labeled_code.yaml
sleep 15
kubectl describe service my-service | grep "External Name:" && echo cloudeval_unit_test_passed
# INCLUDE: "Opening service default/nginx-service in default browser..."
