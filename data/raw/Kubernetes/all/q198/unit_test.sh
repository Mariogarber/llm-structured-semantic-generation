kubectl apply -f labeled_code.yaml

sleep 10

description=$(kubectl describe rs/nginx-replicaset-env)

check_field() {
    local field="$1"
    echo "$description" | grep -P "$field" >/dev/null
    return $?
}

check_field "ENV_VAR:\s+custom_value" && \
check_field "2 desired"

if [ $? -eq 0 ]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml