kubectl apply -f labeled_code.yaml

[ "$(kubectl get secret ssh-secret -o=jsonpath='{.type}')" = "kubernetes.io/ssh-auth" ] && \
echo cloudeval_unit_test_passed
