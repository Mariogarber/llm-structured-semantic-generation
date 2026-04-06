kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=my-ds --timeout=60s

node_selector=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.nodeSelector.kubernetes\.io/os}')
[ "$node_selector" = "linux" ] && \
echo cloudeval_unit_test_passed
