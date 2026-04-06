kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete job/failjob-ignored --timeout=60s

[ "$(kubectl get job failjob-ignored -o=jsonpath='{.status.conditions[*].type}')" = "Complete" ] && \
echo cloudeval_unit_test_passed
