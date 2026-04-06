kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=nginx-daemon --timeout=60s

pods=$(kubectl get pods -l name=nginx-daemon --output=jsonpath={.items..metadata.name})
if kubectl get daemonset | grep "nginx-daemon"; then
    kubectl logs $pods | grep -iq "error" && exit 1 || echo cloudeval_unit_test_passed
fi

