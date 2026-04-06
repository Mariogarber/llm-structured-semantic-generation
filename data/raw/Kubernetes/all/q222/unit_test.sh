kubectl apply -f labeled_code.yaml

cleanup() {
    kubectl delete rolebinding my-role-binding -n default
    kubectl delete role my-role -n default
}

role_permissions=$(kubectl describe role my-role -n default | grep "get")
if [[ $role_permissions == *"pods"* && $role_permissions == *"get"* ]]; then
    echo "role's permissions passed"
else
    cleanup
    echo "Test failed"
fi

role_binding_subject=$(kubectl describe rolebinding my-role-binding -n default | grep "my-user")
if [[ $role_binding_subject == *"User"* && $role_binding_subject == *"my-user"* ]]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

cleanup
