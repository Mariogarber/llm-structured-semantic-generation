kubectl apply -f labeled_code.yaml

sleep 10

description=$(kubectl describe rs/nginx-replicaset-limits)

check_field() {
    local field="$1"
    echo "$description" | grep -P "$field" >/dev/null
    return $?
}

check_field "cpu:\s+500m"  && \
check_field "cpu:\s+250m"  && \
check_field "memory:\s+128Mi" && \
check_field "memory:\s+64Mi" && \
check_field "2 desired"

if [ $? -eq 0 ]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml