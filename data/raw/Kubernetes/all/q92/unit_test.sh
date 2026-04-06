kubectl apply -f labeled_code.yaml
sleep 10
minikube kubectl describe ingress ingress-resource-backend | egrep "Default backend:  APIGroup: k8s.example.com, Kind: StorageBucket, Name: static-assets" && echo cloudeval_unit_test_passed
# INCLUDE: regex like "tomcat-svc:8080 (10.244.0.3:8080)", 10.244.0.3:8080
