kubectl apply -f labeled_code.yaml

quota=$(kubectl describe resourcequota object-counts)
check_field() {
    local field="$1"
    echo "$quota" | grep -P "$field" >/dev/null
    return $?
}

check_field "services\s+[0-9]\s+10" && \
check_field "secrets\s+[0-9]\s+10" && \
check_field "services.loadbalancers\s+[0-9]\s+2"

if [ $? -eq 0 ]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml