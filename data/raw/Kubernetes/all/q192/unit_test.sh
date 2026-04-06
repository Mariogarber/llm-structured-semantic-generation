kubectl apply -f labeled_code.yaml
SELECTOR=$(kubectl get pdb my-pdb -o=jsonpath='{.spec.selector.matchLabels.app}')
UNAVAILABLE=$(kubectl get pdb my-pdb -o=jsonpath='{.spec.maxUnavailable}')

[ "$SELECTOR" = "my-app" ] && \
[ "$UNAVAILABLE" = "1" ] && \
echo cloudeval_unit_test_passed
