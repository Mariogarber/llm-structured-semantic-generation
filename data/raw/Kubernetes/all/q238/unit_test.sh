kubectl apply -f labeled_code.yaml

[ "$(kubectl get secret my-sec -o=jsonpath='{.type}')" = "kubernetes.io/basic-auth" ] && \
[ "$(kubectl get secret my-sec -o=jsonpath='{.data.username}' | base64 -d)" = "harry" ] && \
[ "$(kubectl get secret my-sec -o=jsonpath='{.data.password}' | base64 -d)" = "verysecret" ] && \
echo cloudeval_unit_test_passed
