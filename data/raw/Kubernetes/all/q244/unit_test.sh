kubectl apply -f labeled_code.yaml

kubectl get svc -n kube-system | grep -q "state-metrics" && \
[ "$(kubectl get svc state-metrics -n kube-system -o=jsonpath='{.spec.clusterIP}')" = "None" ] && \
kubectl get svc state-metrics -n kube-system -o=jsonpath='{.spec.ports[*].port}' | grep -q "8080 8081" && \
kubectl get svc state-metrics -n kube-system -o=jsonpath='{.metadata.labels.app\.kubernetes\.io/component}' | grep -q "exporter" && \
kubectl get svc state-metrics -n kube-system -o=jsonpath='{.metadata.labels.app\.kubernetes\.io/version}' | grep -q "2.9.3" && \
echo cloudeval_unit_test_passed