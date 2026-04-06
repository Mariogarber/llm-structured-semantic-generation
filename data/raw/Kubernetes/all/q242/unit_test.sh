kubectl apply -f labeled_code.yaml

kubectl get svc svc-master -o=jsonpath='{.spec.type}' | grep -q "NodePort" && \
kubectl get svc svc-master -o=jsonpath='{.spec.clusterIP}' | grep -q "10.96.0.2" && \
kubectl get svc svc-master -o=jsonpath='{.spec.ports[?(@.name=="redis")].port}' | grep -q "6379" && \
kubectl get svc svc-master -o=jsonpath='{.spec.ports[?(@.name=="metrics")].port}' | grep -q "9121" && \
echo cloudeval_unit_test_passed
