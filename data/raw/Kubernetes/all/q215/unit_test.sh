kubectl apply -f labeled_code.yaml
output=$(kubectl describe clusterrole custom-clusterrole)

echo "$output" | grep "pods" | grep -q "[get list watch]" && \
echo "$output" | grep "nodes" | grep -q "[get list]" && \
echo "$output" | grep "services" | grep -q "[get watch]" && \
echo cloudeval_unit_test_passed
