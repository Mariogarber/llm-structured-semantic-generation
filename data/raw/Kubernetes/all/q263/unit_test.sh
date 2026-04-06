kubectl apply -f labeled_code.yaml

[ "$(kubectl get role pod-reader -o jsonpath='{.rules[0].resources[0]}')" = "pods" ] && \
[ "$(kubectl get role pod-reader -o jsonpath='{.rules[0].verbs[0]}')" = "get" ] && \
[ "$(kubectl get role pod-reader -o jsonpath='{.rules[0].verbs[1]}')" = "list" ] && \
[ "$(kubectl get rolebinding read-pods -o jsonpath='{.subjects[0].name}')" = "ms-account" ] && \
[ "$(kubectl get rolebinding read-pods -o jsonpath='{.roleRef.name}')" = "pod-reader" ] && \
echo cloudeval_unit_test_passed