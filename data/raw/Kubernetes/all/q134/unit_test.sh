kubectl apply -f labeled_code.yaml

[ "$(kubectl get pv pv1 -o jsonpath='{.metadata.annotations.pv\.beta\.kubernetes\.io/gid}')" = "1234" ] &&
echo cloudeval_unit_test_passed
