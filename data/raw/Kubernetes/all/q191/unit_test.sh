kubectl apply -f labeled_code.yaml
SELECTOR=$(kubectl get networkpolicy deny-all -o=jsonpath='{.spec.podSelector.matchLabels.app}')
POLICY_TYPE=$(kubectl get networkpolicy deny-all -o=jsonpath='{.spec.policyTypes[0]}')

[ "$SELECTOR" = "sample-app" ] && \
[ "$POLICY_TYPE" = "Ingress" ] && \
echo cloudeval_unit_test_passed
