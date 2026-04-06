kubectl apply -f labeled_code.yaml

[ "$(kubectl get svc svc-n-ep -o=jsonpath='{.spec.ports[0].port}')" = 80 ] && \
[ "$(kubectl get endpoints svc-n-ep -o=jsonpath='{.subsets[0].addresses[0].ip}')" = "192.0.2.42" ] && \
[ "$(kubectl get endpoints svc-n-ep -o=jsonpath='{.subsets[0].ports[0].port}')" = 9376 ] && \
echo cloudeval_unit_test_passed
