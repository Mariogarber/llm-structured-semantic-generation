kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=fluentd-elasticsearch --timeout=20s
pods=$(kubectl get pods -l name=fluentd-elasticsearch --output=jsonpath={.items..metadata.name})

if kubectl get daemonset | grep "fluentd-elasticsearch"; then
    kubectl logs $pods | grep -i "error" && exit 1 || echo cloudeval_unit_test_passed
fi