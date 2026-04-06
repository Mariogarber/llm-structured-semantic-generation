kubectl apply -f labeled_code.yaml

sleep 5
description=$(kubectl describe resourcequota pods-high)

check_field() {
    local field="$1"
    echo "$description" | grep -P "$field" >/dev/null
    return $?  
}

check_field "cpu\s+0\s+1k" && \
check_field "memory\s+0\s+200Gi" && \
check_field "pods\s+0\s+10"

if [ $? -eq 0 ]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml