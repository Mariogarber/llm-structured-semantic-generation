kubectl apply -f labeled_code.yaml
output=$(kubectl describe clusterrole updated-clusterrole)

echo "$output" | grep "pods" | grep -q "[get list watch]" && \
echo "$output" | grep "services" | grep -q "[get]" && \
echo "$output" | grep "deployments" | grep -q "[get list]" && \
echo "$output" | grep "configmaps" | grep -q "[watch]" && \
echo cloudeval_unit_test_passed
