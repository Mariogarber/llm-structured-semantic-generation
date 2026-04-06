kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=kube-registry-modified --timeout=60s

passed_tests=0
total_tests=3
pods=$(kubectl get pods -l app=kube-registry-modified --output=jsonpath={.items..metadata.name})
host_ip=$(kubectl get pod $pods -o=jsonpath='{.status.hostIP}')
curl_output=$(curl -s -o /dev/null -w "%{http_code}" $host_ip:5000)

if [ "$curl_output" == "200" ]; then
    ((passed_tests++))
else
    exit 1
fi

env_vars=$(kubectl get pods --selector=app=kube-registry-modified -o=jsonpath='{.items[0].spec.containers[0].env[*].name}')
if [[ $env_vars == *"REGISTRY_HOST"* && $env_vars == *"REGISTRY_PORT"* ]]; then
    ((passed_tests++))
fi

cpu_limit=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].resources.limits.cpu}')
memory_limit=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].resources.limits.memory}')
if [ "$cpu_limit" == "100m" ] && [ "$memory_limit" == "50Mi" ]; then
    ((passed_tests++))
fi

if [ $passed_tests -eq $total_tests ]; then
    echo cloudeval_unit_test_passed
fi