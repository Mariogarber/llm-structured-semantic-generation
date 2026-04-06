kubectl apply -f labeled_code.yaml

quota=$(kubectl describe resourcequota object-counts)
check_field() {
    local field="$1"
    echo "$quota" | grep -P "$field" >/dev/null
    return $?
}

check_field "configmaps\s+[0-9]\s+10" && \
check_field "persistentvolumeclaims\s+0\s+4" && \
check_field "pods\s+0\s+4" && \
check_field "replicationcontrollers\s+0\s+20"

if [ $? -eq 0 ]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

kubectl delete -f labeled_code.yaml