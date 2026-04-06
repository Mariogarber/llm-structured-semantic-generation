kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=hostalias-daemonset --timeout=20s
pods=$(kubectl get pods -l app=hostalias-daemonset --output=jsonpath={.items..metadata.name})
if kubectl get daemonset | grep "hostalias-daemonset"; then
    kubectl logs $pods | grep -i "error" && exit 1 || echo cloudeval_unit_test_passed1
    kubectl exec  $pods -- cat /etc/hosts | grep -i "10.1.2.3" && echo cloudeval_unit_test_passed
fi
