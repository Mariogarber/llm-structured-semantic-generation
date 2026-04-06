kubectl apply -f labeled_code.yaml

sleep 10

description=$(kubectl describe rs/nginx-replicaset)

check_field() {
    local field="$1"
    echo "$description" | grep -P "$field" >/dev/null
    return $?
}

check_field "http-get"  && \
check_field ":80"  && \
check_field "Created pod: nginx-replicaset"  && \
check_field "2 desired"

if [ $? -eq 0 ]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml