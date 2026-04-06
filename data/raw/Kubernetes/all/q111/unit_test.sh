kubectl create -f labeled_code.yaml
kubectl create -f labeled_code.yaml

[ "$(kubectl get jobs -l jobgroup=jexample -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep '^kube-job-' | wc -l)" = "2" ] && \
echo cloudeval_unit_test_passed
