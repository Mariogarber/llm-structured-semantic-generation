kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=hellods-modified --timeout=60s
pods=$(kubectl get pods -l app=hellods-modified --output=jsonpath={.items..metadata.name} | awk '{print $NF}')
echo $pods
args_check=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].args}' | grep -o "while true; do sleep 5; done")
echo $args_check
if [[ -z "$args_check" ]]; then
    exit 1
fi

cpu_request=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].resources.requests.cpu}')
memory_request=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].resources.requests.memory}')

if [[ "$cpu_request" != "100m" || "$memory_request" != "200Mi" ]]; then
    exit 1
fi

grace_period=$(kubectl get pod $pods -o=jsonpath='{.spec.terminationGracePeriodSeconds}')
if [[ "$grace_period" -ne 30 ]]; then
    exit 1
fi

echo cloudeval_unit_test_passed
