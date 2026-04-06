kubectl apply -f labeled_code.yaml

[ "$(kubectl get ep myep -o=jsonpath='{.subsets[0].addresses[*].ip}')" = "10.1.2.3 10.1.2.4" ] && \
[ "$(kubectl get ep myep -o=jsonpath='{.subsets[0].ports[0].port}')" = "8080" ] && \
echo cloudeval_unit_test_passed
