kubectl apply -f labeled_code.yaml

sleep 10

description=$(kubectl describe rs/busybox-replicaset)

check_field() {
    local field="$1"
    echo "$description" | grep -P "$field" >/dev/null
    return $?
}

check_field "3600" && \
check_field "Created pod: busybox-replicaset" && \
check_field "2 desired"

if [ $? -eq 0 ]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml