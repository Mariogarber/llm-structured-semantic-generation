kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l component=exporter-modified --timeout=60s

service_endpoints=$(kubectl get services exporter-modified -o=jsonpath='{.spec.selector}' | grep -o '"component":"exporter-modified"')
if [[ -z "$service_endpoints" ]]; then
    exit 1
fi
pods=$(kubectl get pods --selector=component=exporter-modified --output=jsonpath={.items..metadata.name} | awk '{print $NF}')
kubectl describe daemonset exporter-modified | grep "9445/TCP" && echo cloudeval_unit_test_passed