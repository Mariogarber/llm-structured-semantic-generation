kubectl apply -f labeled_code.yaml

[ "$(kubectl get configmap database-config -o=jsonpath='{.data.DB_HOST}')" = "database-host" ] && \
[ "$(kubectl get configmap database-config -o=jsonpath='{.data.DB_PORT}')" = "5432" ] && \
echo cloudeval_unit_test_passed
