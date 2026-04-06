kubectl apply -f labeled_code.yaml
output=$(kubectl describe clusterrole buggy-clusterrole)

echo "$output" | grep "pods" | grep -q "[get list watch]" && \
echo "$output" | grep "services" | grep -q "[get]" && \
echo "$output" | grep "deployments" | grep -q "[get list]" && \
echo "$output" | grep "jobs" | grep -q "[create delete]" && \
echo cloudeval_unit_test_passed
